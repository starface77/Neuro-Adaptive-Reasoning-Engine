#!/usr/bin/env python3
"""Generate predictions.json for SWE-bench submission from VARE results."""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nare.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare.oracle import build_oracle_from_spec


def generate_patch_from_answer(answer: str, task_id: str) -> str:
    """Convert VARE answer (file paths) to a minimal patch format.

    For SWE-bench, we need to generate actual diff patches.
    This is a placeholder - real implementation would need to:
    1. Read the files mentioned in the answer
    2. Generate proper unified diff format
    """
    # For now, create a minimal patch that indicates the files to modify
    lines = answer.strip().split('\n')
    files = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

    if not files:
        return ""

    # Create a minimal patch format
    patch_lines = []
    for filepath in files:
        patch_lines.append(f"--- a/{filepath}")
        patch_lines.append(f"+++ b/{filepath}")
        patch_lines.append("@@ -1,1 +1,1 @@")
        patch_lines.append(f" # Modified by VARE for {task_id}")
        patch_lines.append("")

    return "\n".join(patch_lines)


def run_benchmark_and_generate_predictions(
    tasks_file: str = "benchmarks/swebench_real.json",
    persist_dir: str = "memory_swe_submission",
    output_file: str = "predictions.json",
    log_dir: str = "logs_submission"
):
    """Run full SWE-bench and generate predictions.json."""

    # Load tasks
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    print(f"=" * 80)
    print(f"VARE SWE-bench Submission Generator")
    print(f"=" * 80)
    print(f"Tasks: {len(tasks)}")
    print(f"Model: Claude Sonnet 4.5 via Gemini API")
    print(f"Memory: {persist_dir}")
    print(f"Output: {output_file}")
    print(f"Logs: {log_dir}/")
    print(f"=" * 80)
    print()

    # Create directories
    Path(log_dir).mkdir(exist_ok=True)
    Path(persist_dir).mkdir(exist_ok=True)

    # Initialize agent
    agent = NAREProductionAgent(
        config=DEFAULT_CONFIG,
        persist_dir=persist_dir,
        embedding_dim=3072  # Gemini embeddings
    )

    predictions = []
    stats = {
        "total": 0,
        "completed": 0,
        "errors": 0,
        "routes": {}
    }

    for i, task in enumerate(tasks):
        task_id = task['id']
        query = task['query']
        oracle_spec = task.get('oracle_spec')

        print(f"[{i+1}/{len(tasks)}] {task_id}")

        if not oracle_spec:
            print(f"  SKIP: No oracle spec")
            continue

        # Rate limiting
        if i > 0 and i % 10 == 0:
            import time
            print(f"  [Rate limit pause: 30s]")
            time.sleep(30)

        try:
            # Build oracle
            oracle = build_oracle_from_spec(oracle_spec)

            # Solve
            result = agent.solve(query, oracle=oracle)

            answer = result.get('final_answer', '')
            route = result.get('route_decision', 'UNKNOWN')

            stats["routes"][route] = stats["routes"].get(route, 0) + 1
            stats["total"] += 1

            # Generate patch
            model_patch = generate_patch_from_answer(answer, task_id)

            # Create prediction
            prediction = {
                "instance_id": task_id,
                "model_patch": model_patch,
                "model_name_or_path": "VARE-Claude-Sonnet-4.5"
            }

            predictions.append(prediction)
            stats["completed"] += 1

            print(f"  ✓ route={route:<8} files={len(answer.split())}")

            # Save reasoning log
            log_file = Path(log_dir) / f"{task_id}.txt"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Task: {task_id}\n")
                f.write(f"Route: {route}\n")
                f.write(f"Query:\n{query}\n\n")
                f.write(f"Answer:\n{answer}\n\n")
                f.write(f"Full Result:\n{json.dumps(result, indent=2)}\n")

        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {e}")
            stats["errors"] += 1

            # Log error
            error_file = Path(log_dir) / f"{task_id}.error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Task: {task_id}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())

        # Save intermediate results every 10 tasks
        if (i + 1) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2)
            print(f"  [Checkpoint: {len(predictions)} predictions saved]")

    # Save final predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    # Save stats
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print()
    print(f"=" * 80)
    print(f"COMPLETED")
    print(f"=" * 80)
    print(f"Total: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Errors: {stats['errors']}")
    print(f"Routes: {stats['routes']}")
    print(f"Output: {output_file}")
    print(f"Logs: {log_dir}/")
    print(f"=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SWE-bench predictions")
    parser.add_argument("--tasks", default="benchmarks/swebench_real.json")
    parser.add_argument("--persist-dir", default="memory_swe_submission")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--log-dir", default="logs_submission")

    args = parser.parse_args()

    run_benchmark_and_generate_predictions(
        tasks_file=args.tasks,
        persist_dir=args.persist_dir,
        output_file=args.output,
        log_dir=args.log_dir
    )
