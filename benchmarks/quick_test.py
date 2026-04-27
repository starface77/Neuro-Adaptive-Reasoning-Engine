"""
NARE Quick Test — Sanity Check (6 tasks)
=========================================
Purpose: Verify that all 4 routing levels work correctly.
  - Tasks 1-2: Novel → SLOW (first encounter)
  - Task 3:    Similar → HYBRID (retrieval-augmented)
  - Task 3+:   SLEEP triggers (cluster detected)
  - Task 4:    Post-sleep → REFLEX or HYBRID
  - Task 5:    Exact repeat → FAST cache
  - Task 6:    OOD task → SLOW (no skill matches)

Expected runtime: ~5-8 minutes with Gemma-3-27B free tier.
"""

import sys, os, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NARE_MEMORY_DIR"] = os.path.join(os.path.dirname(__file__), "memory_store")

from nare.agent import NAREProductionAgent

TASKS = [
    # --- Cluster: Arithmetic Sequences (should crystallize) ---
    {
        "query": "Find the next number in the sequence: 3, 6, 9, 12, 15. What comes next?",
        "expected": "18",
        "domain": "arithmetic_sequence",
    },
    {
        "query": "What is the next term in this sequence: 10, 20, 30, 40, 50?",
        "expected": "60",
        "domain": "arithmetic_sequence",
    },
    {
        "query": "Continue the pattern: 7, 14, 21, 28. What is the next number?",
        "expected": "35",
        "domain": "arithmetic_sequence",
    },
    # --- Post-sleep: should use REFLEX or HYBRID ---
    {
        "query": "Find the next number: 4, 8, 12, 16, 20. What comes after?",
        "expected": "24",
        "domain": "arithmetic_sequence",
    },
    # --- Exact repeat: should hit FAST cache ---
    {
        "query": "Find the next number in the sequence: 3, 6, 9, 12, 15. What comes next?",
        "expected": "18",
        "domain": "arithmetic_sequence",
    },
    # --- Out-of-distribution: should go SLOW ---
    {
        "query": "A train leaves Moscow at 9:00 AM traveling at 80 km/h. Another train leaves from the same station at 11:00 AM traveling at 120 km/h in the same direction. At what time will the second train catch up to the first?",
        "expected": "3:00 PM",  # flexible match
        "domain": "word_problem",
    },
]


def check_answer(result: str, expected: str) -> bool:
    """Flexible answer checking: expected substring in result."""
    result_clean = result.lower().strip().replace(",", "")
    expected_clean = expected.lower().strip().replace(",", "")
    # Direct containment
    if expected_clean in result_clean:
        return True
    # Try numeric extraction
    import re
    nums_result = re.findall(r'-?\d+\.?\d*', result_clean)
    nums_expected = re.findall(r'-?\d+\.?\d*', expected_clean)
    if nums_expected and nums_result:
        return nums_expected[0] in nums_result
    return False


def main():
    print("=" * 60)
    print("   NARE Quick Test — Sanity Check (6 tasks)")
    print("=" * 60)

    agent = NAREProductionAgent()
    results = []

    for i, task in enumerate(TASKS):
        print(f"\n{'─'*60}")
        print(f"[Task {i+1}/{len(TASKS)}] {task['query'][:80]}...")
        print(f"  Domain: {task['domain']}  |  Expected: {task['expected']}")

        agent.wait_for_sleep()

        start = time.time()
        res = agent.solve(task["query"])
        elapsed = time.time() - start

        route = res["route_decision"]
        answer = res["final_answer"]
        correct = check_answer(answer, task["expected"])

        results.append({
            "task_id": i + 1,
            "domain": task["domain"],
            "route": route,
            "correct": correct,
            "time": round(elapsed, 2),
            "answer_preview": answer[:100],
        })

        status = "✓ CORRECT" if correct else "✗ WRONG"
        print(f"  Route:  {route}")
        print(f"  Answer: {answer[:120]}")
        print(f"  Status: {status}")
        print(f"  Time:   {elapsed:.2f}s")

    # Wait for any remaining background sleep
    agent.wait_for_sleep()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("   RESULTS SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    routes_seen = set(r["route"] for r in results)

    print(f"\n  Accuracy:   {correct_count}/{total} ({100*correct_count/total:.0f}%)")
    print(f"  Routes hit: {', '.join(sorted(routes_seen))}")
    print()

    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} Task {r['task_id']:2d} | {r['route']:20s} | {r['time']:7.2f}s | {r['domain']}")

    # Route distribution
    print(f"\n  Route Distribution:")
    from collections import Counter
    route_counts = Counter(r["route"] for r in results)
    for route, count in sorted(route_counts.items()):
        print(f"    {route:20s}: {count} tasks ({100*count/total:.0f}%)")

    # Check all route types appeared
    expected_routes = {"SLOW"}  # At minimum SLOW should appear
    missing = expected_routes - routes_seen
    if missing:
        print(f"\n  ⚠ Missing expected routes: {missing}")
    if len(routes_seen) >= 3:
        print(f"\n  ✓ System activated {len(routes_seen)} different routes — routing works!")
    else:
        print(f"\n  ⚠ Only {len(routes_seen)} route(s) seen. Run full_benchmark.py for deeper test.")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
