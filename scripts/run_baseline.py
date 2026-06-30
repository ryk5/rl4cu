"""
Baseline evaluation: run Qwen2.5-Coder-32B-Instruct zero-shot on KernelBench
problems and evaluate the generated kernels on Modal H100s.

Usage:
    python scripts/run_baseline.py --level 1 --limit 20
    python scripts/run_baseline.py --level 1 --level 2 --limit 100

Results are saved to results/baseline_<model>_level<N>_<timestamp>.jsonl
"""

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# project root on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from rl4cu.env.kernelbench import load_problems

load_dotenv()

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"  # via openai-compatible endpoint
# for zero-shot baseline we'll call via openai api pointing at a provider
# that hosts qwen2.5-coder-32b — or set OPENAI_BASE_URL to your own endpoint

RESULTS_DIR = pathlib.Path("results/baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_kernel(client: OpenAI, messages: list[dict], model: str) -> str:
    """Call the LLM and extract the kernel code from the response."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # greedy for baseline
        max_tokens=4096,
    )
    content = response.choices[0].message.content or ""
    return _extract_code(content)


def _extract_code(text: str) -> str:
    """Pull out the first ```python ... ``` block, or return raw text."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return text.strip()


def compute_fast_p(results: list[dict], threshold: float) -> float:
    """fast_p = fraction of problems that are correct AND speedup >= threshold."""
    if not results:
        return 0.0
    hits = sum(
        1 for r in results
        if r.get("correct") and r.get("speedup", -1) >= threshold
    )
    return hits / len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, action="append", default=[], help="KernelBench level(s) to eval (1, 2, 3)")
    parser.add_argument("--limit", type=int, default=None, help="Max problems per level")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL — falls back to OPENAI_BASE_URL env var")
    parser.add_argument("--dry-run", action="store_true", help="Generate kernels but skip Modal eval")
    args = parser.parse_args()

    levels = args.level or [1]

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or None
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
    )

    # lazy import modal only when actually evaluating
    if not args.dry_run:
        import modal as _modal
        eval_kernel = _modal.Function.from_name("rl4cu-kernel-eval", "eval_kernel")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model.replace("/", "_").replace(".", "_")

    all_results = []

    for level in levels:
        print(f"\n=== Level {level} ===")
        problems = load_problems(level=level)
        if args.limit:
            problems = problems[: args.limit]

        print(f"loaded {len(problems)} problems")

        level_results = []
        kernel_jobs = []  # (problem, kernel_code) pairs, submit all at once

        # --- generation pass ---
        print("generating kernels...")
        for problem in tqdm(problems):
            messages = problem.format_prompt()
            try:
                kernel_code = generate_kernel(client, messages, args.model)
            except Exception as e:
                print(f"  problem {problem.problem_id}: generation failed — {e}")
                kernel_code = ""

            kernel_jobs.append((problem, kernel_code))

        if args.dry_run:
            print("dry run — skipping eval")
            for problem, kernel_code in kernel_jobs:
                level_results.append({
                    "problem_id": problem.problem_id,
                    "level": level,
                    "kernel_code": kernel_code,
                    "compiled": None,
                    "correct": None,
                    "speedup": -1.0,
                    "error_msg": None,
                })
        else:
            # --- eval pass: submit all to Modal in parallel ---
            print(f"submitting {len(kernel_jobs)} evals to Modal H100s...")
            inputs = [
                {
                    "kernel_code": kernel_code,
                    "ref_code": problem.ref_code,
                    "problem_id": problem.problem_id,
                }
                for problem, kernel_code in kernel_jobs
            ]

            # .starmap unpacks each dict as kwargs
            modal_results = list(
                eval_kernel.starmap(
                    [(i["kernel_code"], i["ref_code"], i["problem_id"]) for i in inputs]
                )
            )

            for (problem, kernel_code), modal_result in zip(kernel_jobs, modal_results):
                level_results.append({
                    "problem_id": problem.problem_id,
                    "level": level,
                    "name": problem.name,
                    "kernel_code": kernel_code,
                    **modal_result,
                })

        # save level results
        out_path = RESULTS_DIR / f"baseline_{model_slug}_level{level}_{timestamp}.jsonl"
        with open(out_path, "w") as f:
            for r in level_results:
                f.write(json.dumps(r) + "\n")

        # print summary
        n = len(level_results)
        compiled = sum(1 for r in level_results if r.get("compiled"))
        correct  = sum(1 for r in level_results if r.get("correct"))
        fast_0   = compute_fast_p(level_results, 0.0)
        fast_1   = compute_fast_p(level_results, 1.0)
        fast_1_2 = compute_fast_p(level_results, 1.2)

        print(f"\nlevel {level} results ({n} problems):")
        print(f"  compiled:   {compiled}/{n} ({100*compiled/n:.1f}%)")
        print(f"  correct:    {correct}/{n}  ({100*correct/n:.1f}%)")
        print(f"  fast_0:     {100*fast_0:.1f}%  (correct)")
        print(f"  fast_1:     {100*fast_1:.1f}%  (correct + >=1x speedup)")
        print(f"  fast_1.2:   {100*fast_1_2:.1f}%  (correct + >=1.2x speedup)")
        print(f"  saved to:   {out_path}")

        all_results.extend(level_results)

    # combined summary if multiple levels
    if len(levels) > 1:
        n = len(all_results)
        print(f"\n=== Combined ({n} problems) ===")
        print(f"  fast_0:   {100*compute_fast_p(all_results, 0.0):.1f}%")
        print(f"  fast_1:   {100*compute_fast_p(all_results, 1.0):.1f}%")
        print(f"  fast_1.2: {100*compute_fast_p(all_results, 1.2):.1f}%")


if __name__ == "__main__":
    main()
