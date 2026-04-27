"""
NARE Full Benchmark — 24 Real Tasks, 6 Domains
=================================================
Purpose: Comprehensive evaluation of the NARE cognitive cycle.

Domain clusters (tasks within each cluster are structurally similar,
which should trigger sleep/crystallization after ~3 tasks):

  A. Arithmetic Sequences      (4 tasks) — should crystallize into REFLEX
  B. Text Extraction            (4 tasks) — should crystallize into REFLEX
  C. Math Word Problems         (4 tasks) — diverse, mostly SLOW/HYBRID
  D. Basic Math / Formulas      (4 tasks) — should partially crystallize
  E. String / Logic Puzzles     (4 tasks) — diverse reasoning
  F. Out-of-Distribution        (4 tasks) — novel, should stay SLOW

Tasks are interleaved to simulate realistic usage patterns.
Each task has a verifiable expected answer.

Expected runtime: ~30-60 minutes with Gemma-3-27B free tier.
"""

import sys, os, time, json, re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NARE_MEMORY_DIR"] = os.path.join(os.path.dirname(__file__), "memory_store")

from nare.agent import NAREProductionAgent


# ═══════════════════════════════════════════════════════════
#   TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════

TASKS = [
    # ── A. Arithmetic Sequences ──────────────────────────
    {
        "id": 1,
        "query": "Find the next number in the arithmetic sequence: 5, 10, 15, 20, 25. What comes next?",
        "expected": "30",
        "domain": "A_arithmetic_seq",
    },
    {
        "id": 2,
        "query": "What is the next term in this sequence: 3, 7, 11, 15, 19?",
        "expected": "23",
        "domain": "A_arithmetic_seq",
    },
    {
        "id": 3,
        "query": "Continue the arithmetic pattern: 12, 24, 36, 48. What is the next number?",
        "expected": "60",
        "domain": "A_arithmetic_seq",
    },
    # ── B. Text Extraction (first batch) ─────────────────
    {
        "id": 4,
        "query": "Extract all email addresses from this text: 'Contact us at support@example.com or sales@company.org for more info. CC: admin@test.net'",
        "expected": "support@example.com",
        "domain": "B_text_extraction",
    },
    {
        "id": 5,
        "query": "Extract all email addresses from: 'Send your resume to hr@jobs.io and a copy to manager@jobs.io. Questions? help@jobs.io'",
        "expected": "hr@jobs.io",
        "domain": "B_text_extraction",
    },
    {
        "id": 6,
        "query": "Find all email addresses in this message: 'Dear team, please forward to alice@corp.com and bob@corp.com. Best, carol@corp.com'",
        "expected": "alice@corp.com",
        "domain": "B_text_extraction",
    },
    # ── A. Arithmetic Sequences (post-sleep) ─────────────
    {
        "id": 7,
        "query": "Find the next number in the sequence: 8, 16, 24, 32, 40. What comes after 40?",
        "expected": "48",
        "domain": "A_arithmetic_seq",
    },
    # ── C. Math Word Problems ────────────────────────────
    {
        "id": 8,
        "query": "A store sells apples for $2 each. If you buy 15 apples and pay with a $50 bill, how much change do you get back?",
        "expected": "20",
        "domain": "C_word_problem",
    },
    {
        "id": 9,
        "query": "A rectangle has a length of 12 cm and a width of 5 cm. What is its area in square centimeters?",
        "expected": "60",
        "domain": "C_word_problem",
    },
    {
        "id": 10,
        "query": "If a car travels at 60 km/h for 2.5 hours, what is the total distance traveled in kilometers?",
        "expected": "150",
        "domain": "C_word_problem",
    },
    {
        "id": 11,
        "query": "A shirt costs $40. It is on sale for 25% off. What is the sale price in dollars?",
        "expected": "30",
        "domain": "C_word_problem",
    },
    # ── B. Text Extraction (post-sleep) ──────────────────
    {
        "id": 12,
        "query": "Extract all email addresses from: 'Invitations sent to john@party.com, jane@party.com, and mike@party.com.'",
        "expected": "john@party.com",
        "domain": "B_text_extraction",
    },
    # ── D. Basic Math / Formulas ─────────────────────────
    {
        "id": 13,
        "query": "What is the factorial of 7? (i.e., 7!)",
        "expected": "5040",
        "domain": "D_math_formula",
    },
    {
        "id": 14,
        "query": "Calculate the sum of all integers from 1 to 100.",
        "expected": "5050",
        "domain": "D_math_formula",
    },
    {
        "id": 15,
        "query": "What is 2 raised to the power of 10?",
        "expected": "1024",
        "domain": "D_math_formula",
    },
    {
        "id": 16,
        "query": "What is the greatest common divisor (GCD) of 48 and 18?",
        "expected": "6",
        "domain": "D_math_formula",
    },
    # ── E. String / Logic Puzzles ────────────────────────
    {
        "id": 17,
        "query": "Is the word 'racecar' a palindrome? Answer YES or NO.",
        "expected": "YES",
        "domain": "E_logic",
    },
    {
        "id": 18,
        "query": "How many vowels (a, e, i, o, u) are in the word 'encyclopedia'?",
        "expected": "6",
        "domain": "E_logic",
    },
    {
        "id": 19,
        "query": "What is the reverse of the string 'hello world'?",
        "expected": "dlrow olleh",
        "domain": "E_logic",
    },
    {
        "id": 20,
        "query": "Given the list [3, 1, 4, 1, 5, 9, 2, 6, 5], what is the maximum value?",
        "expected": "9",
        "domain": "E_logic",
    },
    # ── F. Out-of-Distribution (novel) ───────────────────
    {
        "id": 21,
        "query": "Convert the Roman numeral MCMXCIV to a decimal (Arabic) number.",
        "expected": "1994",
        "domain": "F_ood",
    },
    {
        "id": 22,
        "query": "How many days are there in the months of January, February (non-leap year), and March combined?",
        "expected": "90",
        "domain": "F_ood",
    },
    {
        "id": 23,
        "query": "In a class of 30 students, 18 play football and 14 play basketball. If 6 play both sports, how many students play neither?",
        "expected": "4",
        "domain": "F_ood",
    },
    {
        "id": 24,
        "query": "A palindrome number reads the same forwards and backwards. What is the largest 3-digit palindrome number that is divisible by 7?",
        "expected": "994",  # Hmm, let me recalculate... 999/7=142.7, 989/7=141.3, 979/7=139.9, 969/7=138.4, 959/7=137, so 959. Actually let me check: 959/7=137. Yes, 959.
        "domain": "F_ood",
    },
]

# Fix task 24 expected answer
TASKS[-1]["expected"] = "959"


def check_answer(result: str, expected: str) -> bool:
    """Flexible answer matching."""
    r = result.lower().strip().replace(",", "").replace("$", "")
    e = expected.lower().strip().replace(",", "").replace("$", "")

    # Direct containment
    if e in r:
        return True

    # Numeric comparison
    nums_r = re.findall(r'-?\d+\.?\d*', r)
    nums_e = re.findall(r'-?\d+\.?\d*', e)
    if nums_e and nums_r:
        for ne in nums_e:
            if ne in nums_r:
                return True
        # Float comparison
        try:
            target = float(nums_e[0])
            for nr in nums_r:
                if abs(float(nr) - target) < 0.01:
                    return True
        except ValueError:
            pass

    return False


def main():
    print("=" * 70)
    print("   NARE Full Benchmark — 24 Tasks, 6 Domains")
    print("=" * 70)

    agent = NAREProductionAgent()
    results = []
    domain_stats = {}

    for i, task in enumerate(TASKS):
        print(f"\n{'─'*70}")
        print(f"[Task {task['id']}/{len(TASKS)}] ({task['domain']})")
        print(f"  Q: {task['query'][:90]}{'...' if len(task['query'])>90 else ''}")

        agent.wait_for_sleep()

        start = time.time()
        res = agent.solve(task["query"])
        elapsed = time.time() - start

        route = res["route_decision"]
        answer = res["final_answer"]
        correct = check_answer(answer, task["expected"])

        results.append({
            "task_id": task["id"],
            "domain": task["domain"],
            "route": route,
            "correct": correct,
            "time": round(elapsed, 2),
            "expected": task["expected"],
            "got": answer[:150],
        })

        # Per-domain tracking
        d = task["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"correct": 0, "total": 0, "routes": [], "times": []}
        domain_stats[d]["total"] += 1
        domain_stats[d]["routes"].append(route)
        domain_stats[d]["times"].append(elapsed)
        if correct:
            domain_stats[d]["correct"] += 1

        status = "✓" if correct else "✗"
        print(f"  Route:    {route}")
        print(f"  Expected: {task['expected']}")
        print(f"  Got:      {answer[:120]}")
        print(f"  Result:   {status}  |  Time: {elapsed:.2f}s")

    # Wait for any remaining background sleep
    agent.wait_for_sleep()

    # ═══════════════════════════════════════════════════════
    #   FINAL REPORT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("   FINAL REPORT")
    print(f"{'='*70}")

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    total_time = sum(r["time"] for r in results)

    print(f"\n  Overall Accuracy:  {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"  Total Time:        {total_time:.1f}s")
    print(f"  Avg Time/Task:     {total_time/total:.1f}s")

    # ── Per-Domain Breakdown ──
    print(f"\n  {'Domain':<22} {'Acc':>6}  {'Routes':>30}  {'Avg Time':>10}")
    print(f"  {'─'*22} {'─'*6}  {'─'*30}  {'─'*10}")
    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        acc = f"{s['correct']}/{s['total']}"
        routes_str = ", ".join(s["routes"])
        avg_t = sum(s["times"]) / len(s["times"])
        print(f"  {domain:<22} {acc:>6}  {routes_str:>30}  {avg_t:>9.1f}s")

    # ── Route Distribution ──
    print(f"\n  Route Distribution:")
    route_counts = Counter(r["route"] for r in results)
    for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {route:<22} {count:>3} ({pct:5.1f}%) {bar}")

    # ── Amortization Analysis ──
    print(f"\n  Amortization Analysis:")
    slow_times = [r["time"] for r in results if r["route"] == "SLOW"]
    fast_times = [r["time"] for r in results if r["route"] in ("FAST", "REFLEX", "REFLEX_PROVISIONAL")]
    hybrid_times = [r["time"] for r in results if r["route"] == "HYBRID"]

    if slow_times:
        print(f"    SLOW  avg: {sum(slow_times)/len(slow_times):>8.2f}s  (n={len(slow_times)})")
    if hybrid_times:
        print(f"    HYBRID avg: {sum(hybrid_times)/len(hybrid_times):>7.2f}s  (n={len(hybrid_times)})")
    if fast_times:
        print(f"    FAST/REFLEX avg: {sum(fast_times)/len(fast_times):>5.2f}s  (n={len(fast_times)})")
        if slow_times:
            speedup = (sum(slow_times)/len(slow_times)) / (sum(fast_times)/len(fast_times) + 0.001)
            print(f"    Speedup (SLOW → REFLEX): {speedup:,.0f}×")

    # ── Token Savings Estimate ──
    reflex_count = sum(1 for r in results if r["route"] in ("FAST", "REFLEX", "REFLEX_PROVISIONAL"))
    if reflex_count:
        print(f"\n  Token Savings:")
        print(f"    Tasks bypassing LLM: {reflex_count}/{total} ({100*reflex_count/total:.0f}%)")
        est_tokens_saved = reflex_count * 700  # ~700 tokens per SLOW call
        print(f"    Est. tokens saved:   ~{est_tokens_saved:,}")

    # ── Per-Task Detail Table ──
    print(f"\n  {'ID':>3} {'Domain':<22} {'Route':<22} {'Time':>8} {'OK':>3}  {'Expected':>10} → {'Got':<40}")
    print(f"  {'─'*3} {'─'*22} {'─'*22} {'─'*8} {'─'*3}  {'─'*10}   {'─'*40}")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        got_short = r["got"][:40].replace("\n", " ")
        print(f"  {r['task_id']:>3} {r['domain']:<22} {r['route']:<22} {r['time']:>7.2f}s  {mark}  {r['expected']:>10} → {got_short:<40}")

    # ── Final Semantic Rules ──
    print(f"\n  Final Skill Registry:")
    if agent.memory.semantic_rules:
        for rule in agent.memory.semantic_rules:
            name = rule.get("pattern", "?")
            conf = rule.get("confidence", 0)
            mat = rule.get("maturity", 0)
            reuse = rule.get("reuse_rate", 0)
            print(f"    [{name}] conf={conf:.2f}, maturity={mat}, reuses={reuse}")
    else:
        print(f"    (none crystallized)")

    print(f"\n{'='*70}")

    # Save results to JSON
    report_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": total,
            "correct": correct_count,
            "accuracy_pct": round(100 * correct_count / total, 1),
            "total_time_s": round(total_time, 1),
            "route_distribution": dict(route_counts),
            "domain_stats": {k: {"correct": v["correct"], "total": v["total"]} for k, v in domain_stats.items()},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {report_path}")


if __name__ == "__main__":
    main()
