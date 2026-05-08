#!/usr/bin/env python3
"""SWE-bench benchmark for NARE/VARE system."""

import json
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare.oracle import build_oracle_from_spec


def run_swe_bench(tasks_file: str = "benchmarks/swebench_real.json", persist_dir: str = "memory_swe"):
    """Run SWE-bench benchmark."""

    # Load tasks
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} SWE-bench tasks")
    print(f"Using: Full NARE system")
    print(f"Memory: {persist_dir}")
    print()

    # Initialize agent
    agent = NAREProductionAgent(
        config=DEFAULT_CONFIG,
        persist_dir=persist_dir,
        embedding_dim=3072  # Gemini embeddings
    )

    results = []
    correct = 0
    total = 0
    route_counts = {}

    for i, task in enumerate(tasks):
        task_id = task['id']
        query = task['query']
        oracle_spec = task.get('oracle_spec')

        print(f"[{i+1}/{len(tasks)}] Task: {task_id}")

        if not oracle_spec:
            print("  SKIP: No oracle spec")
            continue

        # Rate limiting: wait between tasks to avoid API limits
        if i > 0:
            time.sleep(5)  # 5 seconds between tasks

        # Build oracle
        oracle = build_oracle_from_spec(oracle_spec)

        t0 = time.time()
        try:
            result = agent.solve(query, oracle=oracle)
            elapsed = time.time() - t0

            answer = result.get('final_answer', '')
            route = result.get('route_decision', '?')
            alpha = result.get('alpha', 0.0)

            route_counts[route] = route_counts.get(route, 0) + 1

            # Check answer
            ok, info = oracle(query, answer)

            total += 1
            if ok:
                correct += 1
                print(f"  PASS  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s")
            else:
                info_str = info if isinstance(info, str) else str(info)
                print(f"  FAIL  route={route:<8} sim={alpha:.3f}  {elapsed:6.2f}s  {info_str[:80]}")

            results.append({
                'task_id': task_id,
                'correct': ok,
                'route': route,
                'alpha': alpha,
                'elapsed_s': round(elapsed, 2),
                'answer': answer[:200]
            })

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                'task_id': task_id,
                'correct': False,
                'route': 'ERROR',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    # Summary
    accuracy = correct / total if total > 0 else 0
    amortization = sum(route_counts.get(r, 0) for r in ("FAST", "REFLEX", "COMPILED_SKILL"))
    amortization_pct = (amortization / total * 100) if total > 0 else 0

    print()
    print("=" * 60)
    print(f"NARE SWE-bench Results: {correct}/{total} ({accuracy:.1%})")
    print(f"Routing: {route_counts}")
    print(f"Amortization: {amortization_pct:.1f}% ({amortization}/{total})")
    print("=" * 60)

    # Save results
    output = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'system': 'NARE-full',
        'routing': route_counts,
        'amortization_pct': amortization_pct,
        'results': results
    }

    out_file = f"results_swe_{persist_dir}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='benchmarks/swebench_real.json')
    parser.add_argument('--persist-dir', default='memory_swe')
    args = parser.parse_args()

    run_swe_bench(args.tasks, args.persist_dir)
