# rl4cu

using reinforcement learning to teach llms how to write fast cuda kernels. inspired by hls4ml. the idea is that llms are actually pretty bad at writing optimized gpu code out of the box — they can get something that compiles and is maybe correct, but rarely something that's actually fast. so i'm using execution-based rl with rewards tied to correctness + speedup vs pytorch baseline to push the model to learn what actually makes a kernel good.

benchmarking against [kernelbench (stanford, icml 2025)](https://arxiv.org/abs/2502.10517) — 250 problems ranging from single ops to full model architectures. infra runs on modal so i can parallelize kernel compilation + evaluation across many h100s at once.

---

## results so far

**zero-shot baseline — qwen2.5-coder-32b-instruct, kernelbench level 1 (20 problems)**

| metric | value |
|---|---|
| compiled | 14/20 (70%) |
| correct | 10/20 (50%) |
| fast_1 (correct + faster than pytorch) | 0/20 (0%) |
| avg speedup on correct kernels | ~0.14x |

so yeah — 50% of kernels are correct, but every single one is slower than pytorch. the best i saw was 0.64x (still 1.6× slower). the worst was 0.001x — a kernel that took 1800ms for something pytorch does in 2.6ms. that's the gap i'm trying to close with rl.

---

## related work

two papers are most relevant:

- **[kevin](https://arxiv.org/abs/2507.11948)** (cognition ai, july 2025) — multi-turn grpo on cuda kernels using qwq-32b. trains on every refinement turn individually, summarizes cots across turns to avoid context blowup. 180 kernelbench tasks. reward = correctness + correctness × speedup. key limitation: uses standard grpo which has a biased advantage estimator in multi-turn settings (current sample is included in its own group mean baseline).

- **[dr. kernel](https://arxiv.org/abs/2602.05885)** (hkust, feb 2026) — fixes kevin's bias issue with trloo (turn-level reinforce leave-one-out), which excludes the current sample from the baseline. also adds profiling-based rewards (pr) and rejection sampling (prs) to prevent "lazy optimization" — the model learning to speed up tiny unimportant sub-ops instead of the actual bottleneck. triton only, qwen3-14b, cold-start sft on 8k gpt-5 trajectories before rl. beats claude-4.5-sonnet and gpt-5 on kernelbench l2 (31.6% fast_1.2).

**the gap i'm targeting**: combining trloo + profiling-aware rewards on *cuda* (not triton), at 32b scale, with a cold-start sft phase and a progressive reward curriculum. none of the prior work does all of this together.

---

## design choices

### base model
- **qwen2.5-coder-32b-instruct** — code-specialized, apache 2.0, strong on systems-level code
- why not qwq-32b (what kevin uses): qwq is reasoning-tuned but not code-specialized. nobody has compared them directly on cuda kernel tasks — worth ablating later
- why not qwen3-14b (what dr. kernel uses): i want to go bigger at 32b and stay on a code-specialized model

### fine-tuning strategy
- **phase 1**: lora (rank=16, targets: q/v/o projections) — fast iteration, fits on 2×h100-80gb with tensor parallelism
- **phase 2**: full fine-tune with deepspeed zero-3 on modal (4–8×h100) starting from the best lora checkpoint

### rl algorithm
- **core**: grpo — no critic/value model needed, ~50% less gpu memory vs ppo, proven for sparse binary rewards
- **multi-turn**: up to 4 refinement turns, compiler errors + correctness failures + profiling summaries fed back as context each turn
- **advantage estimation**: **trloo** (dr. kernel's fix) instead of standard grpo — excludes current sample from group mean, unbiased in multi-turn settings. `A = (N/(N-1)) * (G_i - Ḡ)`
- **training framework**: verl + vllm — what kernelgym (dr. kernel's env) is built on, handles decoupled rollout/training workers across multi-gpu setups

### three things i'm doing differently vs prior work

**1. trloo on cuda (not triton)**
dr. kernel shows trloo fixes biased advantage estimation in multi-turn settings, but only demonstrates it on triton. kevin does cuda but uses standard biased grpo. i'm combining the two — trloo applied to cuda kernel rl.

**2. profiling-aware reward on cuda**
dr. kernel's pr/prs (reward the model for targeting the actual runtime bottleneck, not trivial sub-ops) is triton-only. i'm porting this to cuda using nsight compute metrics. the reward is scaled by `PR = T_generated / T_total` — if the kernel you optimized was 2% of total runtime, you don't get much credit for speeding it up.

**3. progressive reward curriculum**
neither kevin nor dr. kernel does staged training. i start with a curriculum that unlocks reward components one at a time:
- stage 1: reward for syntactically valid cuda
- stage 2: + reward for successful nvcc compilation
- stage 3: + reward for matching pytorch reference output
- stage 4: + reward for speedup over pytorch baseline

jumping straight to sparse speedup rewards causes the model to thrash early when almost everything fails. the curriculum keeps a training signal alive throughout.

**additionally**: cold-start sft before any rl, same as dr. kernel — generate multi-turn trajectories using a strong teacher (gpt-4o), sft on those first, then rl on top.

### reward function

```
R = R_compile + R_correct + R_speed + R_profiling
```

| component | value | condition |
|---|---|---|
| `R_compile` | +0.1 | compiles with nvcc |
| `R_correct` | +0.4 | output matches pytorch reference |
| `R_speed` | -0.1 to +0.5 | log-scaled speedup, only if correct |
| `R_profiling` | 0 to +0.1 | pr weighting, only if correct |

speed reward is log2-scaled so 2x→+0.25, 4x→+0.5 (capped). linear reward would make the model lazy about large speedups.

profiling reward scales down credit if the optimized kernel wasn't the bottleneck — prevents lazy optimization.

anti-reward-hacking via kernelbench's `kernel_static_checker.py` — catches cached computation reuse, input modification, non-default cuda streams.

### infra
- **modal** for all gpu work — kernel eval, rl rollouts, training
- each kernel candidate evaluated in its own modal container (h100), parallel via `.map()`
- kernelbench submodule baked into the modal image at deploy time

---

## todos

### phase 1 — environment + baseline ✓
- [x] scaffold repo — folder structure, pyproject.toml, deps
- [x] add kernelbench as git submodule, install editable
- [x] write modal kernel eval worker — compile → static check → correctness → benchmark
- [x] run qwen2.5-coder-32b zero-shot on kernelbench l1, get baseline numbers
- [x] implement + unit test the full reward function (all 4 components + curriculum)
- [x] integrate kernelbench static checker for anti-cheat

### phase 2 — cold-start sft
- [ ] run full baseline on all 100 l1 problems + 100 l2 problems (not just 20)
- [ ] generate synthetic multi-turn cuda trajectories using gpt-4o on l1/l2 problems
- [ ] filter trajectories — keep only rollouts where at least one turn is correct + fast
- [ ] sft qwen2.5-coder-32b on these trajectories with lora

### phase 3 — grpo-lora training
- [ ] set up verl with qwen2.5-coder-32b + lora (rank=16, q/v/o projections)
- [ ] implement trloo advantage estimation (drop-in over grpo group mean)
- [ ] implement progressive curriculum: stage 1→2→3→4 reward unlocking
- [ ] single-turn grpo on kernelbench l1, validate reward increases + no collapse
- [ ] add multi-turn loop (up to 4 turns) with compiler error + profiling feedback
- [ ] hook up nsight compute for pr_ratio metric
- [ ] train on l1 + l2 combined

### phase 4 — full fine-tune + eval
- [ ] switch lora → full fine-tune with deepspeed zero-3 on modal (4–8×h100)
- [ ] eval on held-out l2 subset — target: beat kevin-32b and dr. kernel-14b on fast_1.2
- [ ] full kernelbench eval on all 3 levels
- [ ] ablations: trloo vs grpo, with/without profiling reward, with/without curriculum, with/without cold-start sft
- [ ] maybe submit to kernelsseum leaderboard
