"""
SFT dataset formatter.

Converts raw trajectory JSONL (from generate_trajectories.py) into
training-ready formats:

  1. messages.jsonl  — OpenAI-style multi-turn conversation records,
                       one record per (problem, turn) where the turn
                       produced a correct kernel. This is the generic
                       format compatible with TRL SFTTrainer and most
                       fine-tuning frameworks.

  2. verl_sft.parquet — VERL SFT format (prompt + response columns),
                        where prompt is the conversation up to the last
                        user message and response is the assistant reply.

Strategy: for each trajectory we keep every turn where the kernel was
correct, not just the best turn. This gives us more data and teaches
the model both first-attempt correctness and refinement behavior.

Usage:
    python scripts/format_sft_dataset.py \\
        --input data/trajectories/trajectories_l1_*.jsonl \\
        --output data/sft/

    python scripts/format_sft_dataset.py \\
        --input data/trajectories/*.jsonl \\
        --min-speedup 0.5 \\
        --best-only        # only keep best turn per trajectory
"""

import argparse
import glob
import json
import pathlib
import sys
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── helpers ───────────────────────────────────────────────────────────────────

def load_trajectories(paths: list[str]) -> list[dict]:
    trajs = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajs.append(json.loads(line))
    return trajs


def turns_to_keep(trajectory: dict, min_speedup: float, best_only: bool) -> list[int]:
    """
    Return indices of turns that should become SFT examples.

    Rules:
    - Turn must be correct
    - Speedup must be >= min_speedup
    - If best_only: only the best (highest speedup) correct turn
    """
    turns = trajectory["turns"]
    candidates = [
        i for i, t in enumerate(turns)
        if t["eval"].get("correct", False)
        and t["eval"].get("speedup", -1.0) >= min_speedup
    ]
    if not candidates:
        return []
    if best_only:
        best = max(candidates, key=lambda i: turns[i]["eval"].get("speedup", -1.0))
        return [best]
    return candidates


def messages_for_turn(trajectory: dict, turn_idx: int) -> list[dict]:
    """
    Return the full conversation messages up to and including the
    assistant response at turn_idx.

    Each turn snapshot already has the full conversation up to the
    assistant reply for that turn, so we just use it directly.
    """
    return trajectory["turns"][turn_idx]["messages"]


def split_prompt_response(messages: list[dict]) -> tuple[list[dict], str]:
    """
    Split messages into (prompt_messages, response_text).
    prompt_messages: everything up to but not including the last assistant message
    response_text: the last assistant message content
    """
    # find last assistant message
    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx is None:
        raise ValueError("no assistant message found")

    prompt = messages[:last_asst_idx]
    response = messages[last_asst_idx]["content"]
    return prompt, response


# ── formatters ────────────────────────────────────────────────────────────────

def to_messages_record(trajectory: dict, turn_idx: int) -> dict:
    """
    Generic multi-turn record with OpenAI messages format.
    Compatible with TRL SFTTrainer (conversations field).
    """
    messages = messages_for_turn(trajectory, turn_idx)
    turn = trajectory["turns"][turn_idx]
    return {
        "messages": messages,
        "problem_id": trajectory["problem_id"],
        "level": trajectory["level"],
        "name": trajectory["name"],
        "turn_idx": turn_idx,
        "speedup": turn["eval"].get("speedup", -1.0),
        "compiled": turn["eval"].get("compiled", False),
        "correct": turn["eval"].get("correct", False),
    }


def to_verl_record(trajectory: dict, turn_idx: int) -> dict:
    """
    VERL MultiTurnSFTDataset format.

    Uses a `messages` column containing the full conversation list.
    VERL trains on ALL assistant turns within the conversation using
    smart loss masking (only assistant responses contribute to loss).

    Required fields per VERL's sft_trainer_engine.yaml:
      messages:    list of {role, content} dicts
      data_source: dataset identifier string
    """
    messages = messages_for_turn(trajectory, turn_idx)
    turn = trajectory["turns"][turn_idx]
    return {
        "messages": messages,
        "data_source": f"rl4cu_kernelbench_l{trajectory['level']}",
        "problem_id": trajectory["problem_id"],
        "level": trajectory["level"],
        "turn_idx": turn_idx,
        "speedup": turn["eval"].get("speedup", -1.0),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Input trajectory JSONL file(s) or glob")
    parser.add_argument("--output", type=str, default="data/sft", help="Output directory")
    parser.add_argument("--min-speedup", type=float, default=0.5,
                        help="Min speedup for a turn to be included (default 0.5)")
    parser.add_argument("--best-only", action="store_true",
                        help="Only keep best turn per trajectory (default: all correct turns)")
    parser.add_argument("--no-parquet", action="store_true", help="Skip parquet output")
    args = parser.parse_args()

    # expand globs
    input_paths = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_paths.extend(expanded)
        elif pathlib.Path(pattern).exists():
            input_paths.append(pattern)
        else:
            print(f"warning: no files matched {pattern!r}")

    if not input_paths:
        print("error: no input files found")
        sys.exit(1)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading trajectories from {len(input_paths)} file(s)...")
    trajs = load_trajectories(input_paths)
    print(f"  loaded {len(trajs)} trajectories")

    # build records
    messages_records = []
    verl_records = []
    skipped = 0

    for traj in trajs:
        keep_indices = turns_to_keep(traj, args.min_speedup, args.best_only)
        if not keep_indices:
            skipped += 1
            continue
        for turn_idx in keep_indices:
            messages_records.append(to_messages_record(traj, turn_idx))
            verl_records.append(to_verl_record(traj, turn_idx))

    print(f"  kept {len(messages_records)} training examples from {len(trajs) - skipped} trajectories")
    print(f"  skipped {skipped} trajectories (no turn met min_speedup={args.min_speedup})")

    if not messages_records:
        print("no records to write, exiting")
        sys.exit(0)

    # speedup stats
    speedups = [r["speedup"] for r in messages_records]
    fast_1 = sum(1 for s in speedups if s >= 1.0)
    fast_1_2 = sum(1 for s in speedups if s >= 1.2)
    print(f"\ntraining set quality:")
    print(f"  total examples:  {len(messages_records)}")
    print(f"  fast_1  (>=1x):  {fast_1} ({fast_1/len(messages_records):.0%})")
    print(f"  fast_1.2(>=1.2x):{fast_1_2} ({fast_1_2/len(messages_records):.0%})")
    print(f"  avg speedup:     {sum(speedups)/len(speedups):.3f}x")
    print(f"  max speedup:     {max(speedups):.3f}x")

    # level breakdown
    by_level = {}
    for r in messages_records:
        lvl = r["level"]
        by_level.setdefault(lvl, []).append(r["speedup"])
    for lvl, spds in sorted(by_level.items()):
        print(f"  L{lvl}: {len(spds)} examples, avg {sum(spds)/len(spds):.3f}x")

    # write messages.jsonl
    messages_path = output_dir / "messages.jsonl"
    with open(messages_path, "w") as f:
        for rec in messages_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote: {messages_path}")

    # write verl_sft.jsonl (VERL can read jsonl too)
    verl_path = output_dir / "verl_sft.jsonl"
    with open(verl_path, "w") as f:
        for rec in verl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote: {verl_path}")

    # write parquet if pandas available
    if not args.no_parquet:
        try:
            import pandas as pd
            df = pd.DataFrame(verl_records)
            # messages column is list-of-dicts — parquet handles this natively
            # via pyarrow's nested type support
            parquet_path = output_dir / "verl_sft.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"wrote: {parquet_path}")

            # also write train/val parquet splits
            split_idx = int(len(df) * 0.9)
            df.iloc[:split_idx].to_parquet(output_dir / "verl_sft_train.parquet", index=False)
            df.iloc[split_idx:].to_parquet(output_dir / "verl_sft_val.parquet", index=False)
            print(f"wrote verl train/val parquet splits: {split_idx}/{len(df)-split_idx}")
        except ImportError:
            print("pandas not installed, skipping parquet output")

    # write a small train/val split (90/10)
    split_idx = int(len(messages_records) * 0.9)
    for split_name, records in [("train", messages_records[:split_idx]),
                                  ("val", messages_records[split_idx:])]:
        path = output_dir / f"messages_{split_name}.jsonl"
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    print(f"wrote train/val split: {split_idx}/{len(messages_records)-split_idx} examples")


if __name__ == "__main__":
    main()
