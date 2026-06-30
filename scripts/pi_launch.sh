#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# pi_launch.sh — launch rl4cu GRPO training on a PrimeIntellect 8×A100 node
#
# Usage:
#   bash pi_launch.sh [--steps 1000] [--run-name my_run]
#
# Prerequisites (run pi_setup.sh first):
#   - /opt/verl installed (VERL v0.6.0)
#   - vllm 0.8.5 installed
#   - ~/rollout_data_l1.parquet present (scp'd from local machine)
#   - Modal CLI configured (for reward evals)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── args ──────────────────────────────────────────────────────────────────────
RUN_NAME="qwen25coder32b_grpo_trloo"
MAX_STEPS=1000
MODEL_PATH="${MODEL_PATH:-/root/hf_cache/Qwen2.5-Coder-32B-Instruct}"
ROLLOUT_DATA="${ROLLOUT_DATA:-/root/rollout_data_l1.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/checkpoints/$RUN_NAME}"
GROUP_SIZE=8
LR=1e-6
LORA_RANK=16

while [[ $# -gt 0 ]]; do
  case $1 in
    --steps)       MAX_STEPS="$2"; shift 2 ;;
    --run-name)    RUN_NAME="$2"; shift 2 ;;
    --model)       MODEL_PATH="$2"; shift 2 ;;
    --data)        ROLLOUT_DATA="$2"; shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# ── write reward function ─────────────────────────────────────────────────────
REWARD_PATH="$OUTPUT_DIR/kernel_reward.py"
cat > "$REWARD_PATH" << 'PYEOF'
"""
VERL custom reward function for rl4cu.

VERL calls compute_score() for each rollout with:
  data_source:  str  — e.g. "kernelbench_l1"
  solution_str: str  — the model's generated kernel code
  ground_truth: str  — the reference PyTorch code (ref_code)
  extra_info:   dict — {"problem_id": int, "step": int, ...}

We evaluate the kernel via Modal (remote GPU) and return a scalar reward.
"""

import sys, os
sys.path.insert(0, "/root/rl4cu")

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
    extra_info = extra_info or {}
    step = extra_info.get("step", _step_counter[0])
    problem_id = extra_info.get("problem_id", 0)

    stage = get_curriculum_stage(step, CURRICULUM_SCHEDULE)
    kernel_code = _extract_code(solution_str)

    try:
        result = _get_eval_fn().remote(
            kernel_code=kernel_code,
            ref_code=ground_truth,
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
PYEOF

# ── write TRLOO patch ─────────────────────────────────────────────────────────
TRLOO_PATH="/root/trloo_verl.py"
cat > "$TRLOO_PATH" << 'PYEOF'
"""
TRLOO advantage estimation — patches verl.trainer.ppo.core_algos.compute_advantage.
Auto-applied when imported. Controlled by RL4CU_USE_TRLOO env var.
"""
import os, torch


def trloo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    G = rewards.shape[0]
    if G < 2:
        return rewards - rewards.mean()
    group_sum = rewards.sum()
    loo_means = (group_sum - rewards) / (G - 1)
    advantages = (rewards - loo_means) * (G / (G - 1))
    std = advantages.std()
    if std > eps:
        advantages = advantages / (std + eps)
    return advantages


def patch_verl_grpo():
    if os.environ.get("RL4CU_USE_TRLOO", "0") != "1":
        print("[TRLOO] disabled")
        return
    try:
        import verl.trainer.ppo.core_algos as core_algos
        _orig = core_algos.compute_advantage

        def _trloo_advantage(token_level_rewards, eos_mask, index, adv_estimator, norm_adv_by_std_in_grpo=True):
            if adv_estimator != "grpo":
                return _orig(token_level_rewards, eos_mask, index, adv_estimator, norm_adv_by_std_in_grpo)
            scores = (token_level_rewards * eos_mask).sum(dim=-1)
            advantages = torch.zeros_like(scores)
            for start, end in index:
                advantages[start:end] = trloo_advantages(scores[start:end])
            seq_len = token_level_rewards.shape[-1]
            token_advantages = advantages.unsqueeze(-1).expand(-1, seq_len) * eos_mask
            return token_advantages, scores

        core_algos.compute_advantage = _trloo_advantage
        print("[TRLOO] patched compute_advantage ✓")
    except Exception as e:
        print(f"[TRLOO] patch failed: {e}")

patch_verl_grpo()
PYEOF

# ── validate inputs ───────────────────────────────────────────────────────────
echo "=== rl4cu RL training ==="
echo "run:        $RUN_NAME"
echo "model:      $MODEL_PATH"
echo "data:       $ROLLOUT_DATA"
echo "steps:      $MAX_STEPS"
echo "output:     $OUTPUT_DIR"
echo ""

if [ ! -f "$ROLLOUT_DATA" ]; then
  echo "ERROR: rollout data not found at $ROLLOUT_DATA"
  echo "  scp -P 1234 ~/Documents/rl4cu/rollout_data_l1.parquet root@<IP>:~/rollout_data_l1.parquet"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "WARNING: model dir not found at $MODEL_PATH — will try HF download at runtime"
  MODEL_PATH="Qwen/Qwen2.5-Coder-32B-Instruct"
fi

# ── environment ───────────────────────────────────────────────────────────────
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/opt/verl:/root/rl4cu:/root:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export RL4CU_USE_TRLOO=1
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0

# ── build command ─────────────────────────────────────────────────────────────
CMD=(
  python3 -m verl.trainer.main_ppo

  # algorithm: GRPO with no KL in reward (KL handled as loss term)
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False

  # data
  "data.train_files=$ROLLOUT_DATA"
  "data.val_files=$ROLLOUT_DATA"
  data.train_batch_size=64
  data.max_prompt_length=4096
  data.max_response_length=4096
  "data.tokenizer=$MODEL_PATH"
  data.trust_remote_code=True

  # model
  "actor_rollout_ref.model.path=$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  # sdpa = PyTorch's built-in scaled dot-product attention (no flash-attn needed)
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa

  # actor FSDP — BF16, param offload frees GPU for vLLM during rollout
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False

  # LoRA
  "actor_rollout_ref.model.lora_rank=$LORA_RANK"
  "actor_rollout_ref.model.lora_alpha=$((LORA_RANK * 2))"
  actor_rollout_ref.model.target_modules=all-linear

  # actor training
  "actor_rollout_ref.actor.optim.lr=$LR"
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.001

  # ref model — CPU offload
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1

  # vLLM rollout — TP=4 across 8 GPUs
  # A100 SXM4 80GB: FSDP actor shard ~8GB, vLLM dummy model ~16GB (32B/4 TP)
  # ~56GB free → 0.6 * 80 = 48GB for KV cache (safe with param_offload)
  actor_rollout_ref.rollout.name=vllm
  "actor_rollout_ref.rollout.n=$GROUP_SIZE"
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.mode=sync

  # reward
  "custom_reward_function.path=$REWARD_PATH"
  custom_reward_function.name=compute_score

  # trainer
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  "trainer.total_training_steps=$MAX_STEPS"
  "trainer.default_local_dir=$OUTPUT_DIR"
  trainer.project_name=rl4cu
  "trainer.experiment_name=$RUN_NAME"
  'trainer.logger=["console"]'
  trainer.save_freq=50
  trainer.test_freq=25
)

echo "launching:"
printf '  %s \\\n' "${CMD[@]}"
echo ""

# Run from /opt/verl so Hydra finds config/
cd /opt/verl
exec "${CMD[@]}"
