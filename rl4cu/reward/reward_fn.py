"""
Reward function for rl4cu.

Staged reward: compile → correctness → speed → profiling.
Each stage only activates if the prior stage passes, which prevents the model
from learning fast-but-wrong kernels and gives it a gradient to follow even
when the final speed signal is sparse.

See README for full design rationale.
"""

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Reward weights (easy to tune via config)
# ---------------------------------------------------------------------------

COMPILE_BONUS    = 0.1   # just for compiling
CORRECT_BONUS    = 0.4   # correct output vs pytorch reference
MAX_SPEED_BONUS  = 0.5   # achieved at ~4x speedup (log2 scale)
MAX_PROF_BONUS   = 0.1   # profiling-based bottleneck targeting
SLOW_PENALTY     = -0.1  # correct but slower than pytorch (< 0.8x)


@dataclass
class RewardComponents:
    compile_reward: float = 0.0
    correct_reward: float = 0.0
    speed_reward: float   = 0.0
    prof_reward: float    = 0.0

    @property
    def total(self) -> float:
        return self.compile_reward + self.correct_reward + self.speed_reward + self.prof_reward

    def __repr__(self) -> str:
        return (
            f"RewardComponents(total={self.total:.3f}, "
            f"compile={self.compile_reward:.2f}, "
            f"correct={self.correct_reward:.2f}, "
            f"speed={self.speed_reward:.2f}, "
            f"prof={self.prof_reward:.2f})"
        )


def compute_reward(
    compiled: bool,
    correct: bool,
    speedup: float = -1.0,
    pr_ratio: Optional[float] = None,
    curriculum_stage: int = 4,
) -> RewardComponents:
    """
    Compute the staged reward for a single kernel evaluation result.

    Args:
        compiled:         Whether the kernel compiled successfully.
        correct:          Whether the kernel output matches the pytorch reference.
        speedup:          ref_runtime / kernel_runtime. -1 means not measured.
        pr_ratio:         Profiling ratio = T_generated_kernel / T_total_cuda_runtime.
                          None means profiling wasn't collected (skipped in reward).
        curriculum_stage: Controls which reward components are active.
                          1 = compile only
                          2 = compile + correct
                          3 = compile + correct + speed
                          4 = compile + correct + speed + profiling (full reward)

    Returns:
        RewardComponents with per-component breakdown and .total property.
    """
    r = RewardComponents()

    if not compiled:
        return r  # 0 reward for compile failure — no penalty, avoids too-conservative policy

    r.compile_reward = COMPILE_BONUS

    if curriculum_stage < 2 or not correct:
        return r

    r.correct_reward = CORRECT_BONUS

    if curriculum_stage < 3 or speedup < 0:
        return r

    r.speed_reward = _speed_reward(speedup)

    if curriculum_stage < 4 or pr_ratio is None:
        return r

    r.prof_reward = _profiling_reward(pr_ratio)

    return r


def _speed_reward(speedup: float) -> float:
    """
    Log2-scaled speed reward.

    - < 0.8x (slower than pytorch):   -0.1  — penalize regression
    - 0.8x – 1.0x (roughly same):      0.0  — neutral
    - > 1.0x (faster):                 log2-scaled up to MAX_SPEED_BONUS

    Why log2: 2x speedup = +0.25, 4x = +0.5 (capped).
    Linear reward would make 4x look only 4x better than 1x,
    but log scale means each doubling is equally rewarded — incentivizes
    genuinely large optimizations rather than 1.01x gains.
    """
    if speedup < 0.8:
        return SLOW_PENALTY
    if speedup < 1.0:
        return 0.0
    return min(MAX_SPEED_BONUS, 0.25 * math.log2(speedup))


def _profiling_reward(pr_ratio: float) -> float:
    """
    Profiling-based reward component (ported from Dr. Kernel's PR reward).

    pr_ratio = T_generated_kernel / T_total_cuda_runtime

    If the kernel you optimized was only 3% of total runtime, you get
    3% of MAX_PROF_BONUS even if your speedup was huge. This prevents
    "lazy optimization" — speeding up cheap ops while leaving the
    real bottleneck untouched.

    pr_ratio should be in [0, 1]. Values > 1 are clamped.
    """
    pr_ratio = max(0.0, min(1.0, pr_ratio))
    return MAX_PROF_BONUS * pr_ratio


# ---------------------------------------------------------------------------
# Curriculum stage scheduler
# ---------------------------------------------------------------------------

def get_curriculum_stage(step: int, schedule: Optional[dict] = None) -> int:
    """
    Returns the current curriculum stage based on training step.

    Default schedule (can be overridden via config):
        step    0–99   → stage 1 (compile only)
        step  100–299  → stage 2 (+ correctness)
        step  300–599  → stage 3 (+ speed)
        step  600+     → stage 4 (full, + profiling)

    Args:
        step:     Current training step.
        schedule: Optional dict mapping stage → start_step, e.g.
                  {1: 0, 2: 100, 3: 300, 4: 600}
    """
    if schedule is None:
        schedule = {1: 0, 2: 100, 3: 300, 4: 600}

    stage = 1
    for s, start in sorted(schedule.items()):
        if step >= start:
            stage = s
    return stage


# ---------------------------------------------------------------------------
# Batch helpers for GRPO rollout scoring
# ---------------------------------------------------------------------------

def score_rollouts(
    results: list[dict],
    curriculum_stage: int = 4,
) -> list[float]:
    """
    Score a list of eval results (as returned by modal_app/kernel_eval.py)
    and return a flat list of total reward values.

    Args:
        results:          List of eval result dicts from eval_kernel.remote().
        curriculum_stage: Current curriculum stage.

    Returns:
        List of float rewards, one per result.
    """
    rewards = []
    for r in results:
        rc = compute_reward(
            compiled=r.get("compiled", False),
            correct=r.get("correct", False),
            speedup=r.get("speedup", -1.0),
            pr_ratio=r.get("pr_ratio", None),
            curriculum_stage=curriculum_stage,
        )
        rewards.append(rc.total)
    return rewards
