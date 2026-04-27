"""
NARE Full Benchmark — 24 Real Tasks, 6 Domains
=================================================
Purpose: Comprehensive evaluation of the NARE cognitive cycle.

Structure is designed to demonstrate the full amortization lifecycle
within a SINGLE run:

  PHASE 1 — Learning (tasks 1-15):
    Clusters of 3+ structurally similar tasks per domain.
    First tasks go SLOW, sleep triggers after cluster detected,
    later tasks in same domain should hit HYBRID or REFLEX.

  PHASE 2 — Transfer & OOD (tasks 16-24):
    Post-crystallization tasks from learned domains → should be fast.
    Out-of-distribution tasks → should go SLOW.

Domains:
  A. Arithmetic Sequences      — should crystallize into REFLEX
  B. Text Extraction (email)   — should crystallize into REFLEX
  C. Math Word Problems        — diverse, mostly SLOW/HYBRID
  D. Basic Math / Formulas     — should partially crystallize
  E. String / Logic Puzzles    — diverse reasoning
  F. Out-of-Distribution       — novel, should stay SLOW

Expected runtime: ~25-45 minutes (Gemma-3-27B free tier).
Fresh memory store is used for clean results.
"""

import sys, os, time, json, re, shutil
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a fresh, isolated memory store for this benchmark
BENCH_MEMORY = os.path.join(os.path.dirname(__file__), "memory_store")
os.environ["NARE_MEMORY_DIR"] = BENCH_MEMORY

from nare.agent import NAREProductionAgent


# ═══════════════════════════════════════════════════════════
#   TASK DEFINITIONS — ordered for optimal crystallization
# ═══════════════════════════════════════════════════════════

TASKS = [
    # ══════════════════════════════════════════════════════
    # PHASE 1: LEARNING — dense clusters per domain
    # ══════════════════════════════════════════════════════

    # ── A. Arithmetic Sequences (cluster of 4) ───────────
    # Expected: Tasks 1-3 → SLOW, sleep crystallizes skill,
    #           Task 4 → REFLEX or HYBRID (amortized)
    {
        "id": 1,
        "query": "Find the next number in the arithmetic sequence: 5, 10, 15, 20, 25. What comes next?",
        "expected": "30",
        "domain": "A_arithmetic_seq",
        "phase": "learn",
    },
    {
        "id": 2,
        "query": "What is the next term in this sequence: 3, 7, 11, 15, 19?",
        "expected": "23",
        "domain": "A_arithmetic_seq",
        "phase": "learn",
    },
    {
        "id": 3,
        "query": "Continue the arithmetic pattern: 12, 24, 36, 48. What is the next number?",
        "expected": "60",
        "domain": "A_arithmetic_seq",
        "phase": "learn",
    },
    {
        "id": 4,
        "query": "Find the next number in the sequence: 8, 16, 24, 32, 40. What comes after 40?",
        "expected": "48",
        "domain": "A_arithmetic_seq",
        "phase": "test",  # should be REFLEX or HYBRID after sleep
    },

    # ── B. Text Extraction — emails (cluster of 4) ──────
    # Expected: Tasks 5-7 → SLOW, sleep crystallizes skill,
    #           Task 8 → REFLEX or HYBRID
    {
        "id": 5,
        "query": "Extract all email addresses from this text: 'Contact us at support@example.com or sales@company.org for more info. CC: admin@test.net'",
        "expected": "support@example.com",
        "domain": "B_text_extraction",
        "phase": "learn",
    },
    {
        "id": 6,
        "query": "Extract all email addresses from: 'Send your resume to hr@jobs.io and a copy to manager@jobs.io. Questions? help@jobs.io'",
        "expected": "hr@jobs.io",
        "domain": "B_text_extraction",
        "phase": "learn",
    },
    {
        "id": 7,
        "query": "Find all email addresses in this message: 'Dear team, please forward to alice@corp.com and bob@corp.com. Best, carol@corp.com'",
        "expected": "alice@corp.com",
        "domain": "B_text_extraction",
        "phase": "learn",
    },
    {
        "id": 8,
        "query": "Extract all email addresses from: 'Invitations sent to john@party.com, jane@party.com, and mike@party.com.'",
        "expected": "john@party.com",
        "domain": "B_text_extraction",
        "phase": "test",
    },

    # ── C. Math Word Problems (cluster of 4) ────────────
    {
        "id": 9,
        "query": "A store sells apples for $2 each. If you buy 15 apples and pay with a $50 bill, how much change do you get back?",
        "expected": "20",
        "domain": "C_word_problem",
        "phase": "learn",
    },
    {
        "id": 10,
        "query": "A rectangle has a length of 12 cm and a width of 5 cm. What is its area in square centimeters?",
        "expected": "60",
        "domain": "C_word_problem",
        "phase": "learn",
    },
    {
        "id": 11,
        "query": "If a car travels at 60 km/h for 2.5 hours, what is the total distance traveled in kilometers?",
        "expected": "150",
        "domain": "C_word_problem",
        "phase": "learn",
    },
    {
        "id": 12,
        "query": "A shirt costs $40. It is on sale for 25% off. What is the sale price in dollars?",
        "expected": "30",
        "domain": "C_word_problem",
        "phase": "learn",
    },

    # ── D. Basic Math / Formulas (cluster of 3) ─────────
    {
        "id": 13,
        "query": "What is the factorial of 7? (i.e., 7!)",
        "expected": "5040",
        "domain": "D_math_formula",
        "phase": "learn",
    },
    {
        "id": 14,
        "query": "Calculate the sum of all integers from 1 to 100.",
        "expected": "5050",
        "domain": "D_math_formula",
        "phase": "learn",
    },
    {
        "id": 15,
        "query": "What is 2 raised to the power of 10?",
        "expected": "1024",
        "domain": "D_math_formula",
        "phase": "learn",
    },

    # ══════════════════════════════════════════════════════
    # PHASE 2: TRANSFER & VERIFICATION
    # ══════════════════════════════════════════════════════

    # ── A. Arithmetic (post-crystallization test) ────────
    {
        "id": 16,
        "query": "What is the next number in this pattern: 100, 200, 300, 400?",
        "expected": "500",
        "domain": "A_arithmetic_seq",
        "phase": "verify",  # should be REFLEX
    },
    {
        "id": 17,
        "query": "Find the next term: 6, 12, 18, 24, 30. What comes next?",
        "expected": "36",
        "domain": "A_arithmetic_seq",
        "phase": "verify",
    },

    # ── B. Text Extraction (post-crystallization test) ───
    {
        "id": 18,
        "query": "Extract email addresses from: 'Please email feedback to review@app.dev or bugs@app.dev'",
        "expected": "review@app.dev",
        "domain": "B_text_extraction",
        "phase": "verify",
    },

    # ── E. Logic / String Puzzles (novel) ────────────────
    {
        "id": 19,
        "query": "Is the word 'racecar' a palindrome? Answer YES or NO.",
        "expected": "YES",
        "domain": "E_logic",
        "phase": "novel",
    },
    {
        "id": 20,
        "query": "What is the reverse of the string 'hello world'?",
        "expected": "dlrow olleh",
        "domain": "E_logic",
        "phase": "novel",
    },
    {
        "id": 21,
        "query": "Given the list [3, 1, 4, 1, 5, 9, 2, 6, 5], what is the maximum value?",
        "expected": "9",
        "domain": "E_logic",
        "phase": "novel",
    },

    # ── F. Out-of-Distribution (completely novel) ────────
    {
        "id": 22,
        "query": "Convert the Roman numeral MCMXCIV to a decimal (Arabic) number.",
        "expected": "1994",
        "domain": "F_ood",
        "phase": "novel",
    },
    {
        "id": 23,
        "query": "How many days are there in the months of January, February (non-leap year), and March combined?",
        "expected": "90",
        "domain": "F_ood",
        "phase": "novel",
    },
    {
        "id": 24,
        "query": "In a class of 30 students, 18 play football and 14 play basketball. If 6 play both sports, how many students play neither?",
        "expected": "4",
        "domain": "F_ood",
        "phase": "novel",
    },
]


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
        # Float tolerance
        try:
            target = float(nums_e[0])
            for nr in nums_r:
                if abs(float(nr) - target) < 0.01:
                    return True
        except ValueError:
            pass

    return False


def main():
    # Clean memory for reproducible results
    if os.path.exists(BENCH_MEMORY):
        shutil.rmtree(BENCH_MEMORY)
    os.makedirs(BENCH_MEMORY, exist_ok=True)

    print("=" * 70)
    print("   NARE Full Benchmark — 24 Tasks, 6 Domains")
    print("   (Fresh memory store — full cycle in one run)")
    print("=" * 70)

    agent = NAREProductionAgent()
    results = []
    domain_stats = {}
    phase_stats = {"learn": [], "test": [], "verify": [], "novel": []}

    for i, task in enumerate(TASKS):
        print(f"\n{'─'*70}")
        phase_label = {"learn": "🔵 LEARN", "test": "🟡 TEST", "verify": "🟢 VERIFY", "novel": "🔴 NOVEL"}
        print(f"[Task {task['id']}/{len(TASKS)}] {phase_label.get(task['phase'], '?')} ({task['domain']})")
        print(f"  Q: {task['query'][:90]}{'...' if len(task['query'])>90 else ''}")

        agent.wait_for_sleep()

        start = time.time()
        res = agent.solve(task["query"])
        elapsed = time.time() - start

        route = res.get("route", res.get("route_decision", "UNKNOWN"))
        answer = res["final_answer"]
        correct = check_answer(answer, task["expected"])

        result = {
            "task_id": task["id"],
            "domain": task["domain"],
            "phase": task["phase"],
            "route": route,
            "correct": correct,
            "time": round(elapsed, 2),
            "expected": task["expected"],
            "got": answer[:150],
        }
        results.append(result)

        # Per-domain tracking
        d = task["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"correct": 0, "total": 0, "routes": [], "times": []}
        domain_stats[d]["total"] += 1
        domain_stats[d]["routes"].append(route)
        domain_stats[d]["times"].append(elapsed)
        if correct:
            domain_stats[d]["correct"] += 1

        # Per-phase tracking
        phase_stats[task["phase"]].append(result)

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
    print(f"  Total Time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Avg Time/Task:     {total_time/total:.1f}s")

    # ── Per-Phase Breakdown (key metric!) ──
    print(f"\n  ╔══ AMORTIZATION BY PHASE ════════════════════════════════════╗")
    for phase_name, label in [("learn", "🔵 LEARN (first encounter)"),
                               ("test", "🟡 TEST  (post-crystallization)"),
                               ("verify", "🟢 VERIFY (transfer)"),
                               ("novel", "🔴 NOVEL (out-of-distribution)")]:
        phase_results = phase_stats[phase_name]
        if not phase_results:
            continue
        p_correct = sum(1 for r in phase_results if r["correct"])
        p_total = len(phase_results)
        p_time = sum(r["time"] for r in phase_results) / p_total
        p_routes = Counter(r["route"] for r in phase_results)
        routes_str = ", ".join(f"{r}:{c}" for r, c in p_routes.most_common())
        print(f"  ║ {label}")
        print(f"  ║   Acc: {p_correct}/{p_total}  Avg: {p_time:.1f}s  Routes: {routes_str}")
    print(f"  ╚═══════════════════════════════════════════════════════════════╝")

    # ── Per-Domain Breakdown ──
    print(f"\n  {'Domain':<22} {'Acc':>6}  {'Routes':>35}  {'Avg Time':>10}")
    print(f"  {'─'*22} {'─'*6}  {'─'*35}  {'─'*10}")
    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        acc = f"{s['correct']}/{s['total']}"
        routes_str = ", ".join(s["routes"])
        avg_t = sum(s["times"]) / len(s["times"])
        print(f"  {domain:<22} {acc:>6}  {routes_str:>35}  {avg_t:>9.1f}s")

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
    hybrid_times = [r["time"] for r in results if r["route"] in ("HYBRID",)]

    if slow_times:
        print(f"    SLOW  avg:          {sum(slow_times)/len(slow_times):>8.2f}s  (n={len(slow_times)})")
    if hybrid_times:
        print(f"    HYBRID avg:         {sum(hybrid_times)/len(hybrid_times):>8.2f}s  (n={len(hybrid_times)})")
    if fast_times:
        print(f"    FAST/REFLEX avg:    {sum(fast_times)/len(fast_times):>8.2f}s  (n={len(fast_times)})")
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
    print(f"\n  {'ID':>3} {'Phase':<8} {'Domain':<22} {'Route':<22} {'Time':>8} {'OK':>3}  {'Expected':>10} → {'Got':<40}")
    print(f"  {'─'*3} {'─'*8} {'─'*22} {'─'*22} {'─'*8} {'─'*3}  {'─'*10}   {'─'*40}")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        got_short = r["got"][:40].replace("\n", " ")
        print(f"  {r['task_id']:>3} {r['phase']:<8} {r['domain']:<22} {r['route']:<22} {r['time']:>7.2f}s  {mark}  {r['expected']:>10} → {got_short:<40}")

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
            "domain_stats": {
                k: {"correct": v["correct"], "total": v["total"],
                    "routes": v["routes"]}
                for k, v in domain_stats.items()
            },
            "phase_summary": {
                phase: {
                    "correct": sum(1 for r in rs if r["correct"]),
                    "total": len(rs),
                    "avg_time": round(sum(r["time"] for r in rs) / len(rs), 2) if rs else 0,
                    "routes": dict(Counter(r["route"] for r in rs)),
                }
                for phase, rs in phase_stats.items() if rs
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {report_path}")


if __name__ == "__main__":
    main()
