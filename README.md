# rl4cu

using reinforcement learning to teach llms how to write fast cuda kernels. inspired by hls4ml. the idea is that llms are actually pretty bad at writing optimized gpu code out of the box — they can get something that compiles and is maybe correct, but rarely something that's actually fast. so im using [grpo-style rl](https://arxiv.org/pdf/2402.03300) with execution-based rewards (correctness + speedup vs pytorch baseline) to push the model to learn what makes a kernel actually good.

benchmarking against [kernelbench (stanford, icml 2025)](https://arxiv.org/abs/2502.10517) which includes 250 problems ranging from single ops to full model architectures. infra runs on modal so I can parallelize kernel compilation + evaluation across many gpus at once.

still early, details are in flux 

---

## design choices so far

- Base Model: Qwen2.5-Coder-7B-Instruct as a primary, Qwen2.5-Coder-3B-Instruct as a faster fallback
  - Justification: grpo on 3B has already been demonstrated to match 8B SFT models on Triton kernels, should fit on a single A100-80GB for RL training
  - Maybe use DeepSeek-R1-Distill-Qwen-7B, chain of though reasoning might be useful idk
- RL Algo: GRPO + Multi-Turn
  - Justification: grpo > rpo saves ~50% gpu memory overhead for training (no critic model needed!), simple
- 

---

## todos

- [ ] get the repo scaffolded — folder structure, pyproject, deps
- [ ] set up kernelbench locally and make sure the eval harness runs end to end
- [ ] wire up modal — write the kernel eval worker (compile → correctness check → benchmark)
- [ ] run qwen2.5-coder-32b zero-shot on kernelbench l1/l2 to get baseline numbers (fast_0, fast_1, fast_1.2)
- [ ] implement the reward function — staged: compile bonus → correctness bonus → log-scaled speedup reward
- [ ] add the anti-reward-hacking checks (wrapping kernelbench's static checker)
- [ ] set up verl with qwen2.5-coder-32b + lora (rank 16, q/v/o projections)
- [ ] get single-turn grpo training loop running on kernelbench l1
- [ ] hook up modal rollout eval so all G=16 candidates per problem run in parallel
- [ ] make sure training is stable — reward going up, no collapse
- [ ] add multi-turn refinement — feed compiler errors / wrong outputs back as context, up to 4 turns
- [ ] train on l1 + l2, eval on held-out l2 subset, compare to dr. kernel 14b baseline
- [ ] switch lora → full fine-tune with deepspeed zero-3 on modal (4-8x h100)
- [ ] final kernelbench eval on all 3 levels, maybe submit to kernelsseum leaderboard
- [ ] ablations — single vs multi-turn, reward shaping variants, with/without efficiency reward
- [ ] see if my thing is actually any good lmao
