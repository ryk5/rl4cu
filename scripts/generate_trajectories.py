"""
Cold-start SFT trajectory generation.

For each KernelBench problem, runs a multi-turn loop:
  1. Ask gpt-4o to write a CUDA kernel
  2. Evaluate it on Modal H100
  3. Feed the result back as a refinement prompt
  4. Repeat up to MAX_TURNS times

Keeps rollouts where at least one turn produces a correct kernel.
Saves to data/trajectories/<split>.jsonl in multi-turn conversation format
ready for SFT with VERL / TRL.

Usage:
    python scripts/generate_trajectories.py --level 1 --limit 50
    python scripts/generate_trajectories.py --level 1 --level 2
    python scripts/generate_trajectories.py --level 1 --workers 8  # parallel problems
"""

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Union

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rl4cu.env.kernelbench import KernelProblem, load_problems

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

TEACHER_MODEL = "claude-opus-4-5"
MAX_TURNS = 4          # refinement turns per problem
MAX_TOKENS = 4096      # per generation
TEMPERATURE = 0.8      # some diversity in teacher outputs

# a rollout is "good" if any turn meets ALL of these:
MIN_CORRECT = True     # must be correct
MIN_SPEEDUP = 0.5      # speedup >= 0.5x (relaxed — we want correctness first)

OUTPUT_DIR = pathlib.Path("data/trajectories")

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Strip markdown fences if the model wrapped the code."""
    text = text.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    return text.strip()


def _is_claude(model: str) -> bool:
    return model.startswith("claude")


def call_llm(client, messages: list[dict], model: str) -> Optional[str]:
    """Call OpenAI or Anthropic with exponential backoff on rate limits."""
    wait = 10
    for attempt in range(6):
        try:
            if _is_claude(model):
                # Anthropic: separate system message from conversation
                system = next(
                    (m["content"] for m in messages if m["role"] == "system"), None
                )
                human_messages = [m for m in messages if m["role"] != "system"]
                kwargs = dict(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    messages=human_messages,
                )
                if system:
                    kwargs["system"] = system
                resp = client.messages.create(**kwargs)
                return resp.content[0].text
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=MAX_TOKENS,
                )
                return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err or "overloaded" in err.lower()
            if is_rate_limit and attempt < 5:
                tqdm.write(f"    [rate limit] waiting {wait}s before retry (attempt {attempt+1}/6)...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                tqdm.write(f"    [llm error] {e}")
                return None


def eval_kernel_remote(eval_fn, kernel_code: str, ref_code: str, problem_id: int) -> dict:
    """Call the deployed Modal eval function, return result dict."""
    try:
        return eval_fn.remote(
            kernel_code=kernel_code,
            ref_code=ref_code,
            problem_id=problem_id,
            num_correct_trials=3,
            num_perf_trials=10,
            measure_performance=True,
        )
    except Exception as e:
        return {
            "compiled": False,
            "correct": False,
            "speedup": -1.0,
            "kernel_runtime_ms": -1.0,
            "ref_runtime_ms": -1.0,
            "error_msg": f"modal_error: {e}",
            "problem_id": problem_id,
        }


def is_good_turn(result: dict) -> bool:
    """Whether a single turn meets the quality bar for keeping the trajectory."""
    return (
        result.get("correct", False)
        and result.get("speedup", -1.0) >= MIN_SPEEDUP
    )


# ── core rollout ──────────────────────────────────────────────────────────────

def generate_rollout(
    problem: KernelProblem,
    client,
    eval_fn,
    teacher_model: str = TEACHER_MODEL,
) -> Optional[dict]:
    """
    Run a full multi-turn rollout for one problem.

    Returns a trajectory dict if any turn is good, else None.

    Trajectory format:
    {
        "problem_id": int,
        "level": int,
        "name": str,
        "turns": [
            {
                "messages": [...],    # full conversation up to this turn
                "kernel_code": str,   # extracted code
                "eval": {...},        # modal eval result
            },
            ...
        ],
        "best_turn": int,             # index of best turn (by speedup)
        "any_correct": bool,
        "best_speedup": float,
    }
    """
    messages = problem.format_prompt()
    turns = []

    for turn_idx in range(MAX_TURNS):
        # generate
        raw = call_llm(client, messages, teacher_model)
        if raw is None:
            break  # generation failed, stop this rollout

        kernel_code = extract_code(raw)

        # append assistant response to conversation
        messages = messages + [{"role": "assistant", "content": raw}]

        # evaluate
        result = eval_kernel_remote(eval_fn, kernel_code, problem.ref_code, problem.problem_id)

        turns.append({
            "messages": list(messages),  # snapshot of conversation up to here
            "kernel_code": kernel_code,
            "eval": result,
        })

        # if correct and fast enough, we can stop early (or keep going for more diversity)
        # for now: keep going all MAX_TURNS to get more data
        if turn_idx < MAX_TURNS - 1:
            # add refinement prompt for next turn
            refinement = problem.format_refinement_turn(
                prev_kernel=kernel_code,
                compiled=result.get("compiled", False),
                correct=result.get("correct", False),
                speedup=result.get("speedup", -1.0),
                error_msg=result.get("error_msg"),
            )
            messages = messages + [refinement]

    if not turns:
        return None

    any_correct = any(t["eval"].get("correct", False) for t in turns)

    # need at least one correct turn to keep this trajectory
    if not any_correct:
        return None

    best_turn = max(
        range(len(turns)),
        key=lambda i: turns[i]["eval"].get("speedup", -1.0),
    )

    return {
        "problem_id": problem.problem_id,
        "level": problem.level,
        "name": problem.name,
        "turns": turns,
        "best_turn": best_turn,
        "any_correct": any_correct,
        "best_speedup": turns[best_turn]["eval"].get("speedup", -1.0),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, action="append", default=[], help="KernelBench level(s) (1, 2, 3)")
    parser.add_argument("--limit", type=int, default=None, help="Max problems per level")
    parser.add_argument("--workers", type=int, default=4, help="Parallel problems to process")
    parser.add_argument("--model", type=str, default=TEACHER_MODEL, help="Teacher model")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="Generate kernels but skip Modal eval")
    args = parser.parse_args()

    levels = args.level or [1]

    # build client based on model provider
    model = args.model
    if _is_claude(model):
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    else:
        from openai import OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY_OPENAI") or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_key, base_url="https://api.openai.com/v1")

    if not args.dry_run:
        import modal as _modal
        eval_fn = _modal.Function.from_name("rl4cu-kernel-eval", "eval_kernel")
    else:
        eval_fn = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = pathlib.Path(args.output) if args.output else \
        OUTPUT_DIR / f"trajectories_{'_'.join(f'l{l}' for l in levels)}_{timestamp}.jsonl"

    all_problems = []
    for level in levels:
        problems = load_problems(level=level)
        if args.limit:
            problems = problems[:args.limit]
        all_problems.extend(problems)

    print(f"generating trajectories for {len(all_problems)} problems")
    print(f"teacher: {args.model} | turns: {MAX_TURNS} | workers: {args.workers}")
    print(f"output: {output_path}")

    kept = 0
    total = 0

    with open(output_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_rollout, p, client, eval_fn, args.model): p
                for p in all_problems
            }

            with tqdm(total=len(all_problems), unit="problem") as pbar:
                for future in as_completed(futures):
                    problem = futures[future]
                    total += 1
                    try:
                        trajectory = future.result()
                    except Exception as e:
                        trajectory = None
                        tqdm.write(f"  problem {problem.problem_id}: error — {e}")

                    if trajectory is not None:
                        kept += 1
                        out_f.write(json.dumps(trajectory) + "\n")
                        out_f.flush()
                        tqdm.write(
                            f"  ✓ problem {problem.problem_id} (L{problem.level}) "
                            f"| best speedup: {trajectory['best_speedup']:.3f}x "
                            f"| turns: {len(trajectory['turns'])}"
                        )
                    else:
                        tqdm.write(f"  ✗ problem {problem.problem_id} (L{problem.level}) — no correct turn, dropped")

                    pbar.update(1)
                    pbar.set_postfix(kept=kept, total=total, rate=f"{kept/total:.0%}")

    print(f"\ndone: {kept}/{total} trajectories kept ({kept/total:.0%})")
    print(f"saved to: {output_path}")

    # quick stats
    if kept > 0:
        with open(output_path) as f:
            trajs = [json.loads(l) for l in f]
        speedups = [t["best_speedup"] for t in trajs]
        fast_1 = sum(1 for s in speedups if s >= 1.0)
        fast_1_2 = sum(1 for s in speedups if s >= 1.2)
        print(f"\ntrajectory quality:")
        print(f"  fast_1   (>=1x):   {fast_1}/{kept} ({fast_1/kept:.0%})")
        print(f"  fast_1.2 (>=1.2x): {fast_1_2}/{kept} ({fast_1_2/kept:.0%})")
        print(f"  avg best speedup:  {sum(speedups)/len(speedups):.3f}x")


if __name__ == "__main__":
    main()
