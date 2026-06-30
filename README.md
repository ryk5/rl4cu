# rl4cu

using reinforcement learning to teach llms how to write fast cuda kernels. inspired by hls4ml. the idea is that llms are actually pretty bad at writing optimized gpu code out of the box — they can get something that compiles and is maybe correct, but rarely something that's actually fast. so im using execution-based rl with rewards tied to correctness + speedup vs pytorch baseline to push the model to learn what actually makes a kernel good.

benchmarking against [kernelbench (stanford, icml 2025)](https://arxiv.org/abs/2502.10517) — 250 problems ranging from single ops to full model architectures. infra runs on modal so i can parallelize kernel compilation + evaluation across many gpus at once.

---

## related work

two papers are most relevant:

- **[kevin](https://arxiv.org/abs/2507.11948)** (cognition ai, july 2025) — multi-turn grpo on cuda kernels using qwq-32b. trains on every refinement turn individually, summarizes cots across turns to avoid context blowup. 180 kernelbench tasks. reward = correctness + correctness × speedup. key limitation: uses standard grpo which has a biased advantage estimator in multi-turn settings (current sample included in its own group mean baseline).

- **[dr. kernel](https://arxiv.org/abs/2602.05885)** (hkust, feb 2026) — fixes kevin's bias issue with trloo (turn-level reinforce leave-one-out), which excludes the current sample from the baseline. also adds profiling-based rewards (pr) and rejection sampling (prs) to prevent "lazy optimization" (model learning to speed up tiny unimportant sub-ops instead of the actual bottleneck). triton only, qwen3-14b, cold-start sft on 8k gpt-5 trajectories before rl. beats claude-4.5-sonnet and gpt-5 on kernelbench l2 (31.6% fast_1.2).

**the gap**: i want to combine trloo (dr. kernel recommended this)+ profiling-aware rewards on *cuda* (not triton), at 32b scale (need to avoid lazy optimizations, a naive speedup reward is just the total wall time), with a proper cold-start sft phase and a progressive reward curriculum (dont waste early training steps).

---

## design choices

### base model
- **qwen2.5-coder-32b-instruct** — code-specialized, apache 2.0, strong on systems-level code
- why not qwq-32b (what kevin uses): qwq is reasoning-tuned but not code-specialized. nobody has compared these directly on cuda kernel tasks — worth ablating later
- why not qwen3-14b (what dr. kernel uses): going bigger at 32b to push the ceiling, and staying on a code-specialized model

### fine-tuning strategy
- **phase 1**: lora (rank=16, targets: q/v/o projections) — fast iteration, fits on 2×h100-80gb with tensor parallelism
- **phase 2**: full fine-tune with deepspeed zero-3 on modal (4–8×h100) starting from the best lora checkpoint

### rl algorithm
- **core**: grpo — no critic/value model needed, ~50% less gpu memory vs ppo, proven for sparse binary rewards
- **multi-turn**: up to 4 refinement turns, compiler errors + correctness failures + profiling summaries fed back as context each turn
- **advantage estimation**: **trloo** (dr. kernel's fix) instead of standard grpo — excludes current sample from group mean, unbiased in multi-turn settings. `A = (N/(N-1)) * (G_i - Ḡ)`
- **training framework**: verl + vllm — what kernelgym (dr. kernel's env) is built on, handles decoupled rollout/training workers across multi-gpu setups

### the three novel contributions vs prior work

**1. trloo on cuda (not triton)**
dr. kernel shows trloo fixes biased advantage estimation in multi-turn settings, but only on triton. kevin does cuda but with biased grpo. we're the first to apply trloo to cuda kernel rl.

**2. profiling-aware reward on cuda**
dr. kernel's pr/prs (reward the model for optimizing the actual runtime bottleneck, not trivial sub-ops) is triton-only. we port this to cuda using nsight compute metrics. reward is augmented with `PR = T_generated / T_total` — if the kernel you sped up was 2% of total runtime, you don't get much credit.

**3. progressive reward curriculum**
neither kevin nor dr. kernel does staged training. we start with a curriculum:
- stage 1 (format): reward for producing syntactically valid cuda code
- stage 2 (compile): reward for successful nvcc compilation
- stage 3 (correctness): reward for matching pytorch reference output  
- stage 4 (speed): reward for speedup over pytorch baseline

the idea is that jumping straight to sparse speedup rewards causes the model to thrash early. warming it up through the easier stages first gives a better starting point before the hard signal kicks in.

**additionally**: cold-start sft phase before any rl (following dr. kernel's lead) — generate synthetic multi-turn trajectories using a strong teacher model, sft on those first, then rl on top. neither kevin nor dr. kernel does this for cuda at 32b scale.

### reward function (sort of naive, values somewhat arbitrary)

```
R = R_compile + R_correct + R_speed + R_profiling
```

| component | value | condition |
|---|---|---|
| `R_compile` | +0.1 | code compiles with nvcc |
| `R_correct` | +0.4 | output matches pytorch reference |
| `R_speed` | -0.1 to +0.5 | log-scaled speedup, only if correct |
| `R_profiling` | 0 to +0.1 | PR weighting, only if correct |

speed reward shaping (log-scale to incentivize large wins, not just 1.01x):
```python
def speed_reward(speedup):
    if speedup < 0.8:   return -0.1   # slower than pytorch
    if speedup < 1.0:   return 0.0    # roughly same
    return min(0.5, 0.25 * math.log2(speedup))  # 2x→+0.25, 4x→+0.5
```

profiling reward (penalizes lazy optimization):
```python
def profiling_reward(pr_ratio):
    # pr_ratio = T_generated_kernel / T_total_cuda_runtime
    # if the kernel you optimized was only 5% of runtime, reward is scaled down
    return 0.1 * pr_ratio
```

anti-reward-hacking: wraps kernelbench's `kernel_static_checker.py` — catches cached computation reuse, input modification, non-default cuda streams.

### benchmarking
- **primary**: kernelbench eval harness, `fast_p` metrics (fast_0, fast_1, fast_1.2, fast_2) — directly comparable to [kernelsseum leaderboard](https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/)
- **hardware**: h100 or a100-80gb on modal, `gpu=["H100", "A100-80GB"]` fallback
- **profiling**: nsight compute (`ncu`) on successful kernels for pr metric + occupancy/cache analysis

### infrastructure
- **modal** for all gpu work — kernel compilation, correctness checks, benchmarks, rl rollout evaluation, training itself
- `modal.Function.map()` for parallel candidate evaluation — all G=16 rollouts per problem run simultaneously
- separate modal containers for rollout workers (vllm inference) vs training workers (verl + deepspeed)

---

## todos

### phase 1 — environment + baseline
- [ ] scaffold repo — folder structure, pyproject.toml, deps
- [ ] clone kernelbench, get single kernel eval running end to end
- [ ] write modal kernel eval worker — takes kernel code string, returns `{compile_ok, correct, speedup, error_msg, pr_ratio}`
- [ ] run qwen2.5-coder-32b zero-shot on kernelbench l1 + l2, record fast_0 / fast_1 / fast_1.2 baseline
- [ ] implement + unit test the full reward function (all 4 components)
- [ ] integrate kernelbench static checker for anti-cheat

### phase 2 — cold-start sft
- [ ] generate synthetic multi-turn cuda trajectories using a teacher model (claude / gpt-5) on kernelbench l1/l2 problems
- [ ] filter trajectories for quality (at least one correct + fast turn in the rollout)
- [ ] sft qwen2.5-coder-32b on these trajectories with lora — warm start before rl

### phase 3 — grpo-lora training
- [ ] set up verl with qwen2.5-coder-32b + lora (rank=16, q/v/o projections)
- [ ] implement trloo advantage estimation (drop-in replacement for grpo group mean)
- [ ] implement progressive curriculum: stage 1→2→3→4 reward unlocking
- [ ] single-turn grpo training on kernelbench l1 first, validate reward increases + no collapse
- [ ] add multi-turn loop (up to 4 turns) with compiler error / profiling feedback injection
- [ ] hook up nsight compute for pr_ratio metric
- [ ] train on l1 + l2 combined

### phase 4 — full fine-tune + eval
- [ ] switch lora → full fine-tune with deepspeed zero-3 on modal (4–8×h100)
- [ ] eval on held-out l2 subset — target: beat kevin-32b and dr. kernel-14b on fast_1.2
- [ ] full kernelbench eval on all 3 levels
- [ ] ablations: trloo vs standard grpo, with/without profiling reward, with/without curriculum, with/without cold-start sft
- [ ] maybe submit to kernelsseum leaderboard
