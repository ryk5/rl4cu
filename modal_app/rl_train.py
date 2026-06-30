"""
RL training job on Modal — GRPO with TRLOO advantage estimation.

Trains Qwen2.5-Coder-32B-Instruct (or the SFT-LoRA checkpoint) using VERL's
GRPO implementation with our custom reward function.

Key design choices vs prior work:
  - TRLOO advantage: A_i = (N/(N-1)) * (G_i - mean(G_{-i}))
    Unbiased in multi-turn settings (excludes current sample from baseline)
  - Multi-turn rollout: up to MAX_TURNS refinement turns per problem
  - Progressive curriculum: compile → correct → speed → profiling
  - Modal H100s for both rollout (vLLM) and training (FSDP)

GPU layout on 4xH100:
  - 2 GPUs: vLLM rollout (tensor parallel = 2)
  - 2 GPUs: actor training (FSDP)
  - ref model: param_offload=True to save memory

Usage:
    modal run modal_app/rl_train.py
    modal run modal_app/rl_train.py --run-name ablation_no_trloo --use-trloo=False
    modal run modal_app/rl_train.py --run-name lora_r32 --lora-rank 32
"""

import modal

# ── image ─────────────────────────────────────────────────────────────────────

RL_IMAGE = (
    # official VERL base image: torch 2.6, flash-attn 2.7.4, CUDA 12.4, cuDNN 9.8
    modal.Image.from_registry(
        "verlai/verl:base-verl0.4-cu124-cudnn9.8-torch2.6-fa2.7.4",
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y -qq git",
        # pin to v0.6.0 (first version with sft_trainer + correct ppo_trainer configs)
        "git clone --branch v0.6.0 --depth 1 https://github.com/volcengine/verl /opt/verl"
        " && pip install -e /opt/verl"
        " && python -c 'import verl.trainer.main_ppo; print(\"verl RL OK\")'",
        # vLLM is required for rollout; use 0.8.5 (matches torch 2.6 + CUDA 12.4)
        # Disable v1 engine (VLLM_USE_V1=0) since it pre-allocates KV cache differently
        # and conflicts with the FSDP hybrid engine memory layout
        "VLLM_USE_V1=0 pip install 'vllm==0.8.5'"
        " && VLLM_USE_V1=0 python -c 'import vllm; print(\"vllm OK:\", vllm.__version__)'",
    )
    .pip_install(
        "pandas",
        "pyarrow",
        "wandb",
    )
    .add_local_dir("./rl4cu", remote_path="/root/rl4cu", copy=True)
    .add_local_dir("./kernelbench", remote_path="/root/kernelbench", copy=True)
    .run_commands(
        # install kernelbench without letting it upgrade torch/torchvision
        # (base image has torch 2.6 + torchvision built together; upgrading breaks them)
        "pip install -e /root/kernelbench --no-deps"
        " && pip install"
        " 'litellm[proxy]'"
        " pydra-config"
        " openai"
        " modal",
    )
)

app = modal.App("rl4cu-rl-train")

volume = modal.Volume.from_name("rl4cu-weights", create_if_missing=True)
VOLUME_MOUNT = "/vol"
CHECKPOINTS_DIR = f"{VOLUME_MOUNT}/rl_checkpoints"
SFT_CHECKPOINT_DIR = f"{VOLUME_MOUNT}/sft_checkpoints"
HF_CACHE_DIR = f"{VOLUME_MOUNT}/hf_cache"

# ── config ────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

GROUP_SIZE = 8           # rollouts per problem per step (GRPO n)
MAX_TURNS = 4            # multi-turn refinement turns
LEARNING_RATE = 1e-6
KL_LOSS_COEF = 0.001
CLIP_RATIO = 0.2
PPO_EPOCHS = 1
MAX_PROMPT_LEN = 4096
MAX_RESPONSE_LEN = 4096

# curriculum schedule: step → stage
# stage 1 = compile, 2 = +correct, 3 = +speed, 4 = +profiling
CURRICULUM_SCHEDULE = {1: 0, 2: 50, 3: 150, 4: 400}


# ── reward function ───────────────────────────────────────────────────────────
# VERL's custom reward interface: compute_score(data_source, solution_str, ground_truth, extra_info)
# We write this to disk in the container and pass the path to VERL.

REWARD_FN_CODE = '''\
"""
VERL custom reward function for rl4cu.

VERL calls compute_score() for each rollout with:
  data_source:  str  — e.g. "kernelbench_l1"
  solution_str: str  — the model's generated kernel code
  ground_truth: str  — the reference PyTorch code (ref_code)
  extra_info:   dict — {"problem_id": int, "step": int, ...}

We evaluate the kernel on a Modal H100 and return a scalar reward
using the staged curriculum reward function.
"""

import sys
import os
sys.path.insert(0, "/root")

import modal
from rl4cu.reward.reward_fn import compute_reward, get_curriculum_stage

_eval_fn = None
_step_counter = [0]

CURRICULUM_SCHEDULE = {1: 0, 2: 50, 3: 150, 4: 400}


def _get_eval_fn():
    global _eval_fn
    if _eval_fn is None:
        _eval_fn = modal.Function.from_name("rl4cu-kernel-eval", "eval_kernel")
    return _eval_fn


def _extract_code(text: str) -> str:
    text = text.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> float:
    """
    Evaluate a generated CUDA kernel and return a scalar reward.

    This is called once per rollout by VERL's reward manager.
    Evals are batched by VERL before calling, so this may run in parallel.
    """
    extra_info = extra_info or {}
    step = extra_info.get("step", _step_counter[0])
    problem_id = extra_info.get("problem_id", 0)

    stage = get_curriculum_stage(step, CURRICULUM_SCHEDULE)
    kernel_code = _extract_code(solution_str)
    ref_code = ground_truth

    try:
        eval_fn = _get_eval_fn()
        result = eval_fn.remote(
            kernel_code=kernel_code,
            ref_code=ref_code,
            problem_id=problem_id,
            num_correct_trials=3,
            num_perf_trials=10,
            measure_performance=(stage >= 3),
        )
    except Exception as e:
        print(f"[reward] eval error for problem {problem_id}: {e}")
        return 0.0

    rc = compute_reward(
        compiled=result.get("compiled", False),
        correct=result.get("correct", False),
        speedup=result.get("speedup", -1.0),
        pr_ratio=result.get("pr_ratio", None),
        curriculum_stage=stage,
    )

    _step_counter[0] += 1
    return rc.total
'''


# ── TRLOO patch ───────────────────────────────────────────────────────────────
# Patches VERL's compute_advantage to use leave-one-out baseline.
# Written to disk and imported via PYTHONPATH before training starts.

TRLOO_CODE = '''\
"""
TRLOO (Turn-level REINFORCE Leave-One-Out) advantage estimation.

Patches verl.trainer.ppo.core_algos.compute_advantage to use an
unbiased LOO baseline instead of the standard group mean.

Standard GRPO:  A_i = G_i - mean(G)            # biased: G_i in baseline
TRLOO:          A_i = (N/(N-1)) * (G_i - mean(G_{-i}))  # unbiased

Reference: Dr. Kernel (HKUST, Feb 2026), Section 3.2
"""

import os
import torch


def trloo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute TRLOO advantages for a group of G rollouts.

    Args:
        rewards: shape (G,)
    Returns:
        advantages: shape (G,)
    """
    G = rewards.shape[0]
    if G < 2:
        return rewards - rewards.mean()

    group_sum = rewards.sum()
    loo_means = (group_sum - rewards) / (G - 1)   # leave-one-out mean
    advantages = (rewards - loo_means) * (G / (G - 1))

    std = advantages.std()
    if std > eps:
        advantages = advantages / (std + eps)

    return advantages


def patch_verl_grpo():
    """Monkey-patch VERL compute_advantage to use TRLOO."""
    if not os.environ.get("RL4CU_USE_TRLOO", "0") == "1":
        print("[TRLOO] disabled (RL4CU_USE_TRLOO != 1)")
        return

    try:
        import verl.trainer.ppo.core_algos as core_algos

        _orig = core_algos.compute_advantage

        def _trloo_advantage(token_level_rewards, eos_mask, index, adv_estimator, norm_adv_by_std_in_grpo=True):
            if adv_estimator != "grpo":
                return _orig(token_level_rewards, eos_mask, index, adv_estimator, norm_adv_by_std_in_grpo)

            # token_level_rewards: (batch, seq_len)
            # index: list of (start, end) tuples marking each group
            # aggregate to sequence-level rewards first
            scores = (token_level_rewards * eos_mask).sum(dim=-1)  # (batch,)

            advantages = torch.zeros_like(scores)
            for start, end in index:
                group_rewards = scores[start:end]
                advantages[start:end] = trloo_advantages(group_rewards)

            # expand back to token level
            seq_len = token_level_rewards.shape[-1]
            token_advantages = advantages.unsqueeze(-1).expand(-1, seq_len) * eos_mask
            return token_advantages, scores

        core_algos.compute_advantage = _trloo_advantage
        print("[TRLOO] patched verl.trainer.ppo.core_algos.compute_advantage ✓")

    except Exception as e:
        print(f"[TRLOO] patch failed: {e} — falling back to standard GRPO")


# auto-apply when imported
patch_verl_grpo()
'''


# ── training function ──────────────────────────────────────────────────────────

@app.function(
    image=RL_IMAGE,
    gpu="H100:8",
    timeout=60 * 60 * 12,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    run_name: str = "qwen25coder32b_grpo_trloo",
    actor_checkpoint: str = None,
    group_size: int = GROUP_SIZE,
    max_steps: int = 1000,
    use_trloo: bool = True,
    use_lora: bool = True,
    lora_rank: int = 16,
    learning_rate: float = LEARNING_RATE,
    rollout_data_path: str = None,
):
    import os
    import subprocess

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_dir = f"{CHECKPOINTS_DIR}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # write reward fn and TRLOO patch to disk
    reward_path = f"{output_dir}/kernel_reward.py"
    trloo_path = "/root/trloo_verl.py"
    with open(reward_path, "w") as f:
        f.write(REWARD_FN_CODE)
    with open(trloo_path, "w") as f:
        f.write(TRLOO_CODE)

    # model: use merged SFT checkpoint if it exists, else base model
    if actor_checkpoint:
        model_path = actor_checkpoint
    else:
        merged = f"{SFT_CHECKPOINT_DIR}/merged"
        model_path = merged if os.path.exists(merged) else MODEL_ID

    if rollout_data_path is None:
        rollout_data_path = f"{CHECKPOINTS_DIR}/rollout_data_l1.parquet"

    print(f"model:   {model_path}")
    print(f"data:    {rollout_data_path}")
    print(f"TRLOO:   {use_trloo}")
    print(f"LoRA:    {use_lora} (rank={lora_rank})")
    print(f"G:       {group_size}")
    print(f"steps:   {max_steps}")

    # VERL v0.6.0 PPO/GRPO trainer
    # Config from verl/trainer/config/ppo_trainer.yaml @ v0.6.0
    # Must run from /opt/verl so Hydra finds the config/ directory
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        # algorithm: GRPO
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        # data
        f"data.train_files={rollout_data_path}",
        f"data.val_files={rollout_data_path}",
        # train_batch_size = number of prompts sampled per iteration
        # With 100 L1 problems, must be ≤ 100; using 64 gives plenty of diversity per step
        "data.train_batch_size=64",
        f"data.max_prompt_length={MAX_PROMPT_LEN}",
        f"data.max_response_length={MAX_RESPONSE_LEN}",
        f"data.tokenizer={model_path}",
        "data.trust_remote_code=True",
        # actor model (path comes from model@model: hf_model)
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # use bfloat16 to halve memory vs fp32 default
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        # offload FSDP actor params + optimizer to CPU
        # param_offload: frees ~8GB/GPU shard during rollout
        # optimizer_offload: critical for LoRA — keeps Adam states (~2GB) off GPU at init,
        #   which is when vLLM measures available memory for KV cache allocation
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        # disable torch.compile (avoids CUDA graph memory pools that conflict with vLLM)
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        # LoRA
        *(
            [
                f"actor_rollout_ref.model.lora_rank={lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={lora_rank * 2}",
                "actor_rollout_ref.model.target_modules=all-linear",
            ] if use_lora else []
        ),
        # actor training (from actor/dp_actor.yaml)
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        # ppo_mini_batch_size = mini-batch size for actor update (in tokens)
        # must be ≤ train_batch_size * group_size = 64 * 8 = 512
        f"actor_rollout_ref.actor.ppo_mini_batch_size=64",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.ppo_epochs={PPO_EPOCHS}",
        f"actor_rollout_ref.actor.clip_ratio={CLIP_RATIO}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={KL_LOSS_COEF}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0.001",
        # ref model — offload params to CPU to save GPU memory
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        # rollout (vLLM) — TP=4 on 8-GPU node for the 32B model
        # Memory budget per GPU at KV cache init time (with optimizer_offload=True):
        #   FSDP model shard: ~8GB (32B BF16 / 8 GPUs)
        #   vLLM model (TP=4): ~16GB (32B BF16 / 4 TP)
        #   Remaining: ~56GB → 0.6 × 80GB = 48GB budget, subtract 16GB model = 32GB KV cache
        #   Using 0.4 to be conservative: 0.4 × 80GB = 32GB, minus 16GB model = 16GB KV cache
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={group_size}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        # cap sequence length to control KV cache size; 4096 prompt+response fits our tasks
        f"actor_rollout_ref.rollout.max_model_len={MAX_PROMPT_LEN + MAX_RESPONSE_LEN}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={MAX_PROMPT_LEN + MAX_RESPONSE_LEN}",
        # enforce_eager avoids CUDA graph memory pools (saves ~200MB but more importantly
        # avoids the private pool that was conflicting with vLLM's memory accounting)
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.mode=sync",
        # reward: our custom function (top-level in ppo_trainer.yaml)
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        # trainer
        f"trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        f"trainer.total_training_steps={max_steps}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.project_name=rl4cu",
        f"trainer.experiment_name={run_name}",
        'trainer.logger=["console"]',
        "trainer.save_freq=50",
        "trainer.test_freq=25",
    ]

    print("\nlaunching VERL GRPO:")
    print(" \\\n  ".join(cmd))

    env = os.environ.copy()
    # ensure verl source is on path for all worker processes
    env["PYTHONPATH"] = "/opt/verl:/root:" + env.get("PYTHONPATH", "")
    # disable vLLM v1 engine — it pre-allocates KV cache without considering FSDP memory
    env["VLLM_USE_V1"] = "0"
    # Note: expandable_segments is incompatible with vLLM's memory pool — do not set it
    if use_trloo:
        env["RL4CU_USE_TRLOO"] = "1"

    # run from /opt/verl so Hydra finds the config/ directory
    result = subprocess.run(cmd, check=False, env=env, cwd="/opt/verl")
    if result.returncode != 0:
        raise RuntimeError(f"RL training failed with exit code {result.returncode}")

    volume.commit()
    print(f"\nRL training complete. checkpoints: {output_dir}")
    return output_dir


# ── rollout dataset builder ────────────────────────────────────────────────────

@app.function(
    image=RL_IMAGE,
    cpu=2,
    timeout=60 * 10,
    volumes={VOLUME_MOUNT: volume},
)
def prepare_rollout_dataset(level: int = 1):
    """
    Build a parquet of KernelBench problems as VERL rollout prompts.

    Schema:
      prompt       list[dict]  — OpenAI messages up to first user turn
      ref_code     str         — reference PyTorch implementation
      problem_id   int
      level        int
      data_source  str
    """
    import sys
    import pandas as pd
    sys.path.insert(0, "/root")

    from rl4cu.env.kernelbench import load_problems

    problems = load_problems(level=level)
    records = []
    for p in problems:
        # VERL's legacy_data loader reads 'prompt' as a list of message dicts
        # (same as OpenAI chat format) and applies tokenizer.apply_chat_template
        records.append({
            "prompt": p.format_prompt(),   # list[dict] with role/content keys
            "ref_code": p.ref_code,
            "problem_id": p.problem_id,
            "level": p.level,
            "data_source": f"kernelbench_l{level}",
        })

    import os
    output_path = f"{CHECKPOINTS_DIR}/rollout_data_l{level}.parquet"
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    volume.commit()
    print(f"wrote {len(records)} problems → {output_path}")
    return output_path


# ── local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    run_name: str = "qwen25coder32b_grpo_trloo",
    level: int = 1,
    group_size: int = GROUP_SIZE,
    max_steps: int = 1000,
    use_trloo: bool = True,
    use_lora: bool = True,
    actor_checkpoint: str = None,
):
    print(f"=== rl4cu RL training ===")
    print(f"run:        {run_name}")
    print(f"level:      L{level}")
    print(f"group_size: {group_size}")
    print(f"steps:      {max_steps}")
    print(f"TRLOO:      {use_trloo}")
    print(f"LoRA:       {use_lora}")
    print()

    print("step 1/2: preparing rollout dataset...")
    rollout_data_path = prepare_rollout_dataset.remote(level=level)
    print(f"  → {rollout_data_path}")

    print("step 2/2: starting RL training...")
    output_dir = train.remote(
        run_name=run_name,
        actor_checkpoint=actor_checkpoint,
        group_size=group_size,
        max_steps=max_steps,
        use_trloo=use_trloo,
        use_lora=use_lora,
        rollout_data_path=rollout_data_path,
    )
    print(f"done → {output_dir}")
