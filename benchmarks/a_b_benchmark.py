"""A/B benchmark: NARE  vs  vanilla Gemini (same model, no memory).

Purpose
-------
Quantify how much of NARE's behaviour is attributable to its 4-tier
router + sleep-phase skill induction versus the underlying LLM. The
comparison is **per-task, single-run**:

  * Both arms see the **same prompt** for each task (no FAST cache hit
    from prior tasks counts as "vanilla").
  * Both arms are graded by the **same oracle** built from each task's
    ``oracle_spec`` (so we measure ground-truth correctness, not LLM
    self-judgement).
  * Both arms call Gemma-3-27B-IT through the same `nare.llm` thin
    client, so latency / token cost differences are not infrastructure
    artefacts.

Outputs
-------
  * ``benchmarks/a_b_results.json``   — full per-task records.
  * ``benchmarks/a_b_report.md``      — summary + CIs + per-route breakdown.

Usage
-----
You will run this script. It costs **two LLM calls per task per seed**
(NARE may emit several SLOW samples plus critic calls, vanilla emits
one) — be mindful of your Gemini quota.

  $ export GEMINI_API_KEY=...
  $ python benchmarks/a_b_benchmark.py
  $ python benchmarks/a_b_benchmark.py --num-tasks 30 --num-seeds 3
  $ python benchmarks/a_b_benchmark.py --dataset path/to/tasks.json

Each task in the dataset must be a dict:

  {
    "id": "gsm8k_001",
    "query": "If Alice has 3 apples and Bob gives her 4 more, ...",
    "oracle_spec": {"type": "numeric_set", "expected": [7]},
    # Optional metadata that gets passed through to the report:
    "category": "GSM8K",
    "expected_solution": "7"
  }

The script ships a tiny 15-task built-in set so you can sanity-check
the harness with no extra files. Replace it with your own dataset
(GSM8K / HumanEval-Lite / your benchmark of choice) for a real eval.

Statistics
----------
For accuracy on each arm we report Wilson 95% CIs (no normality
assumption, valid for small n). For the *difference* in accuracy we
report a paired bootstrap 95% CI (10k resamples). Latency is reported
as mean ± 1 std. Amortization (% of NARE answers that hit FAST / REFLEX
/ HYBRID) is reported on the NARE arm only — vanilla has no router.

Caveats
-------
  * Single dataset per run. Mixing GSM8K + HumanEval + Logic in one
    run is fine (the built-in set does this) but per-category accuracy
    is the more honest signal — see the per-category table in the
    report.
  * The "amortization" number depends on whether NARE's memory was
    pre-warmed. By default the script uses an isolated tmp memory dir
    so the first seed pays the cold-start cost. Use ``--persist-dir``
    to reuse a warmed memory store across runs.
  * Vanilla Gemini does not get tool use / code execution. If your
    oracle is ``python_assert`` and the vanilla answer is text rather
    than runnable code, the oracle may reject correct-but-textual
    answers. This is honest: NARE's executability of code is part of
    what we are measuring.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Make `nare` importable when running the script directly from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare import llm  # noqa: E402  (intentional path mutation above)
from nare.agent import NAREProductionAgent  # noqa: E402
from nare.config import DEFAULT_CONFIG  # noqa: E402
from nare.memory import MemorySystem  # noqa: E402
from nare.oracle import build_oracle_from_spec  # noqa: E402


# ---------------------------------------------------------------------
# Built-in mini dataset (15 tasks, 3 categories)
# ---------------------------------------------------------------------

BUILTIN_TASKS: List[Dict[str, Any]] = [
    # ── GSM8K-style arithmetic word problems (5) ────────────────────
    # Each task carries up to 3 ``paraphrases`` — alternate phrasings
    # of the same problem with the same expected answer. The
    # ``--paraphrases N`` CLI flag controls how many of them are
    # actually run. Paraphrases are run sequentially against the same
    # NARE agent so the memory store can warm up: the first paraphrase
    # pays the SLOW cost, the second/third should hit FAST/HYBRID once
    # NARE has induced (or cached) a skill. This is what the project
    # actually claims to deliver.
    {
        "id": "gsm_001",
        "category": "GSM8K",
        "query": (
            "Maria has 3 apples. Bob gives her 4 more. How many apples "
            "does Maria have now?"
        ),
        "paraphrases": [
            "Bob gave Maria 4 apples on top of the 3 she already had. How many apples does Maria have in total?",
            "Maria starts with 3 apples and receives 4 additional apples from Bob. What is her current total?",
        ],
        "expected_solution": "7",
        "oracle_spec": {"type": "numeric_set", "expected": [7]},
    },
    {
        "id": "gsm_002",
        "category": "GSM8K",
        "query": (
            "A train travels at 60 km/h for 2.5 hours. How many "
            "kilometres does it cover?"
        ),
        "paraphrases": [
            "If a train moves at a constant speed of 60 kilometres per hour for two and a half hours, what distance does it travel?",
            "A locomotive maintains 60 km/h for 2.5 h. How far has it gone?",
        ],
        "expected_solution": "150",
        "oracle_spec": {"type": "numeric_set", "expected": [150]},
    },
    {
        "id": "gsm_003",
        "category": "GSM8K",
        "query": (
            "If a shirt costs $20 and is on a 25% discount, what is "
            "the final price in dollars?"
        ),
        "paraphrases": [
            "A $20 shirt is marked down by 25%. How much does it cost after the discount?",
            "Take 25% off a shirt that originally cost $20. What is the resulting price?",
        ],
        "expected_solution": "15",
        "oracle_spec": {"type": "numeric_set", "expected": [15]},
    },
    {
        "id": "gsm_004",
        "category": "GSM8K",
        "query": (
            "A class has 12 boys and 18 girls. What fraction of the "
            "class is boys? Give the answer as a decimal."
        ),
        "paraphrases": [
            "There are 12 boys and 18 girls in a classroom. What proportion of the students are boys, expressed as a decimal?",
            "Out of 12 boys and 18 girls, what decimal fraction of the class is male?",
        ],
        "expected_solution": "0.4",
        "oracle_spec": {"type": "numeric_set", "expected": [0.4]},
    },
    {
        "id": "gsm_005",
        "category": "GSM8K",
        "query": (
            "John has 5 boxes, each containing 8 books. He gives away "
            "12 books. How many books does he have left?"
        ),
        "paraphrases": [
            "John owns 5 boxes of 8 books each. After giving 12 books away, how many books remain?",
            "Start with 5 boxes × 8 books per box. Subtract 12 donated books. What is left?",
        ],
        "expected_solution": "28",
        "oracle_spec": {"type": "numeric_set", "expected": [28]},
    },

    # ── HumanEval-Lite-style code (5) ───────────────────────────────
    # We use string_contains rather than python_assert because vanilla
    # CoT often returns prose with the answer inline rather than a
    # callable function. NARE's executability mode handles both.
    {
        "id": "code_001",
        "category": "HumanEval-Lite",
        "query": "Reverse the string 'hello'.",
        "paraphrases": [
            "Take the string 'hello' and write it backwards.",
            "Output the characters of 'hello' in reverse order.",
        ],
        "expected_solution": "olleh",
        "oracle_spec": {"type": "string_contains", "must_contain": ["olleh"]},
    },
    {
        "id": "code_002",
        "category": "HumanEval-Lite",
        "query": "Reverse the string 'cognition'.",
        "paraphrases": [
            "Print 'cognition' written backwards, character by character.",
            "Apply a string-reversal to 'cognition' and give the result.",
        ],
        "expected_solution": "noitingoc",
        "oracle_spec": {"type": "string_contains", "must_contain": ["noitingoc"]},
    },
    {
        "id": "code_003",
        "category": "HumanEval-Lite",
        "query": "What is the length of the string 'banana'?",
        "paraphrases": [
            "How many characters are in the word 'banana'?",
            "Count the characters in 'banana' and report the count.",
        ],
        "expected_solution": "6",
        "oracle_spec": {"type": "numeric_set", "expected": [6]},
    },
    {
        "id": "code_004",
        "category": "HumanEval-Lite",
        "query": "Is 17 a prime number? Answer yes or no.",
        "paraphrases": [
            "Determine whether 17 is prime. Reply yes or no.",
            "Is the integer 17 a prime? Answer with 'yes' or 'no'.",
        ],
        "expected_solution": "yes",
        "oracle_spec": {"type": "string_contains", "must_contain": ["yes"]},
    },
    {
        "id": "code_005",
        "category": "HumanEval-Lite",
        "query": (
            "Given the list [3, 1, 4, 1, 5, 9, 2, 6], what is the "
            "maximum value?"
        ),
        "paraphrases": [
            "Find the largest element of [3, 1, 4, 1, 5, 9, 2, 6].",
            "Return the maximum number contained in [3, 1, 4, 1, 5, 9, 2, 6].",
        ],
        "expected_solution": "9",
        "oracle_spec": {"type": "numeric_set", "expected": [9]},
    },

    # ── Logic / general QA (5) ──────────────────────────────────────
    {
        "id": "logic_001",
        "category": "Logic",
        "query": (
            "All cats are mammals. Whiskers is a cat. Is Whiskers a "
            "mammal? Answer yes or no."
        ),
        "paraphrases": [
            "Given that every cat is a mammal and Whiskers is a cat, is Whiskers a mammal? Yes or no?",
            "Premise: cats are mammals. Premise: Whiskers is a cat. Conclusion: is Whiskers a mammal? Answer yes/no.",
        ],
        "expected_solution": "yes",
        "oracle_spec": {"type": "string_contains", "must_contain": ["yes"]},
    },
    {
        "id": "logic_002",
        "category": "Logic",
        "query": "What is the capital of France?",
        "paraphrases": [
            "Name the capital city of France.",
            "Which city is the capital of the country France?",
        ],
        "expected_solution": "Paris",
        "oracle_spec": {"type": "string_contains", "must_contain": ["Paris"]},
    },
    {
        "id": "logic_003",
        "category": "Logic",
        "query": "How many sides does a hexagon have?",
        "paraphrases": [
            "A hexagon is a polygon with how many edges?",
            "Count the sides of a hexagon.",
        ],
        "expected_solution": "6",
        "oracle_spec": {"type": "numeric_set", "expected": [6]},
    },
    {
        "id": "logic_004",
        "category": "Logic",
        "query": (
            "If today is Wednesday, what day of the week will it be "
            "in 10 days?"
        ),
        "paraphrases": [
            "Today is a Wednesday. Which weekday will it be 10 days from now?",
            "Starting on Wednesday and counting 10 days forward, what day of the week do you land on?",
        ],
        "expected_solution": "Saturday",
        "oracle_spec": {"type": "string_contains", "must_contain": ["Saturday"]},
    },
    {
        "id": "logic_005",
        "category": "Logic",
        "query": "What is the chemical symbol for gold?",
        "paraphrases": [
            "Which two-letter chemical symbol denotes the element gold?",
            "Gold's chemical symbol on the periodic table is what?",
        ],
        "expected_solution": "Au",
        "oracle_spec": {"type": "string_contains", "must_contain": ["Au"]},
    },
]


# ---------------------------------------------------------------------
# Vanilla-Gemini runner (NO memory, NO critic, NO sleep)
# ---------------------------------------------------------------------

VANILLA_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Think step by step and "
    "then give your final answer on a single line prefixed with "
    "'Final answer:'. Be concise; do not include code blocks unless "
    "explicitly asked."
)


def vanilla_gemini_solve(query: str) -> Tuple[str, float]:
    """Single zero-shot CoT call to Gemma-3-27B-IT.

    Returns (final_answer, latency_seconds). The "Final answer:" line
    is parsed out when present; otherwise the whole response is used.
    A network/quota failure raises — the caller decides how to record
    it.
    """
    t0 = time.time()
    prompt = f"{VANILLA_SYSTEM_PROMPT}\n\nQuestion: {query}"
    # Reuse `generate_samples` so we share the exact same HTTP client,
    # retry logic and rate-limit handling as the NARE arm.
    samples, _tokens = llm.generate_samples(
        prompt, n=1, temperature=0.5, mode="SLOW"
    )
    elapsed = time.time() - t0

    if not samples:
        return "", elapsed

    raw = samples[0].get("solution", "") or samples[0].get("reasoning", "")
    # Heuristic extraction of the "Final answer:" line if present.
    m = re.search(r"final answer\s*[:\-]\s*(.+?)\s*$", raw, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip(), elapsed
    return raw.strip(), elapsed


# ---------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for a proportion. Returns (low, high) as fractions."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap_diff_ci(
    pairs: List[Tuple[bool, bool]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Paired bootstrap 95% CI for (acc_A − acc_B). Returns (point, low, high).

    `pairs` is a list of (A_correct, B_correct) per task. Resamples
    tasks with replacement, computes the accuracy diff each time.
    """
    if not pairs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(pairs)
    diffs: List[float] = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a = sum(1 for a, _ in sample if a) / n
        b = sum(1 for _, b in sample if b) / n
        diffs.append(a - b)
    diffs.sort()
    point = sum(1 for a, _ in pairs if a) / n - sum(1 for _, b in pairs if b) / n
    low = diffs[int(0.025 * n_resamples)]
    high = diffs[int(0.975 * n_resamples) - 1]
    return point, low, high


# ---------------------------------------------------------------------
# Per-task records
# ---------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    category: str
    seed: int
    paraphrase_idx: int  # 0 = original query; 1, 2, ... = i-th paraphrase
    base_task_id: str    # task_id with the paraphrase suffix stripped
    query: str           # the actual query that was sent (so the report
                          # makes sense for paraphrase runs)
    nare_route: str
    nare_answer: str
    nare_correct: bool
    nare_latency_s: float
    nare_oracle_info: str
    vanilla_answer: str
    vanilla_correct: bool
    vanilla_latency_s: float
    vanilla_oracle_info: str
    expected_solution: str

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------

def _build_isolated_agent(persist_dir: Optional[str]) -> NAREProductionAgent:
    """An agent backed by either a caller-supplied persist_dir (warm)
    or a fresh tmp dir (cold-start each seed)."""
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        agent = NAREProductionAgent(config=DEFAULT_CONFIG, persist_dir=persist_dir)
    else:
        # Caller wants a clean slate; we leak the tmp dir to the OS and
        # let the process exit clean it up. Production-style cleanup
        # would track and remove this, but for a one-shot benchmark
        # leakage is harmless.
        tmp = tempfile.mkdtemp(prefix="nare_ab_")
        agent = NAREProductionAgent(config=DEFAULT_CONFIG, persist_dir=tmp)
    return agent


def _grade(answer: str, oracle_spec: Dict[str, Any], query: str) -> Tuple[bool, str]:
    """Run an oracle on ``answer`` for ``query``. Return (correct, info)."""
    try:
        oracle = build_oracle_from_spec(oracle_spec)
    except Exception as e:  # noqa: BLE001
        return False, f"bad oracle_spec: {e}"
    try:
        ok, info = oracle(query, answer)
    except Exception as e:  # noqa: BLE001
        return False, f"oracle crashed: {type(e).__name__}: {e}"
    return bool(ok), str(info)


def _expand_paraphrases(
    tasks: List[Dict[str, Any]], paraphrases: int
) -> List[Dict[str, Any]]:
    """Return a flat list of (task, paraphrase_idx, query) tuples
    expanded so that for each base task we run the original query
    followed by up to ``paraphrases - 1`` paraphrased versions.

    Order matters: paraphrases of the same base task are emitted
    consecutively so that the NARE memory store warms up between them.
    With ``paraphrases=1`` the behaviour is identical to the legacy
    "one query per task" mode.
    """
    expanded: List[Dict[str, Any]] = []
    for task in tasks:
        # paraphrase_idx=0 is always the original query.
        variants: List[str] = [task["query"]]
        variants.extend(task.get("paraphrases", []) or [])
        # Cap at ``paraphrases`` and at the actual number we have.
        n = min(paraphrases, len(variants))
        for idx in range(n):
            inst = dict(task)
            inst["base_task_id"] = task["id"]
            inst["task_id"] = f"{task['id']}#p{idx}"
            inst["paraphrase_idx"] = idx
            inst["query"] = variants[idx]
            expanded.append(inst)
    return expanded


def run_a_b(
    tasks: List[Dict[str, Any]],
    seeds: List[int],
    persist_dir: Optional[str],
    inter_call_sleep_s: float = 0.0,
    paraphrases: int = 1,
) -> List[TaskResult]:
    """Run both arms across every (seed, task) pair. Returns flat result list.

    When ``paraphrases > 1``, each base task is expanded into multiple
    queries (the original plus up to ``paraphrases - 1`` paraphrases
    from the task's ``"paraphrases"`` field). They are run sequentially
    against the **same** NARE agent within a seed, so the memory store
    can warm up — this is the regime where NARE's amortization can
    actually manifest. Vanilla still runs as a single zero-shot CoT
    call per query.
    """
    all_results: List[TaskResult] = []
    expanded = _expand_paraphrases(tasks, paraphrases)

    for seed_i, seed in enumerate(seeds):
        random.seed(seed)
        # Fresh agent per seed unless the caller wants persistent memory
        # across seeds (warm cache).
        agent = _build_isolated_agent(persist_dir)

        print(f"\n=== SEED {seed_i + 1}/{len(seeds)}  (rng={seed}) ===")
        for task_i, task in enumerate(expanded):
            tid = task["task_id"]
            base_tid = task.get("base_task_id", task["task_id"])
            p_idx = task.get("paraphrase_idx", 0)
            cat = task.get("category", "uncategorised")
            query = task["query"]
            spec = task["oracle_spec"]
            expected = task.get("expected_solution", "")

            print(f"  [{task_i + 1}/{len(expanded)}] {tid} ({cat})")

            # ---- NARE arm ----
            t0 = time.time()
            try:
                nare_out = agent.solve(query, oracle_spec=spec)
                nare_answer = nare_out.get("final_answer", "")
                nare_route = nare_out.get("route_decision", "?")
            except Exception as e:  # noqa: BLE001
                nare_answer = ""
                nare_route = f"ERROR:{type(e).__name__}"
                print(f"    NARE  : crashed - {e}")
            nare_lat = time.time() - t0
            nare_ok, nare_info = _grade(nare_answer, spec, query)
            print(f"    NARE  : {'PASS' if nare_ok else 'FAIL'}  route={nare_route:<8}  {nare_lat:6.2f}s  {nare_info}")

            if inter_call_sleep_s > 0:
                time.sleep(inter_call_sleep_s)

            # ---- Vanilla arm ----
            try:
                vanilla_answer, vanilla_lat = vanilla_gemini_solve(query)
            except Exception as e:  # noqa: BLE001
                vanilla_answer, vanilla_lat = "", 0.0
                print(f"    VAN.  : crashed - {e}")
            vanilla_ok, vanilla_info = _grade(vanilla_answer, spec, query)
            print(f"    VAN.  : {'PASS' if vanilla_ok else 'FAIL'}  {vanilla_lat:6.2f}s  {vanilla_info}")

            all_results.append(
                TaskResult(
                    task_id=tid,
                    category=cat,
                    seed=seed,
                    paraphrase_idx=p_idx,
                    base_task_id=base_tid,
                    query=query[:500],
                    nare_route=nare_route,
                    nare_answer=nare_answer[:500],
                    nare_correct=nare_ok,
                    nare_latency_s=nare_lat,
                    nare_oracle_info=nare_info,
                    vanilla_answer=vanilla_answer[:500],
                    vanilla_correct=vanilla_ok,
                    vanilla_latency_s=vanilla_lat,
                    vanilla_oracle_info=vanilla_info,
                    expected_solution=expected,
                )
            )

            if inter_call_sleep_s > 0:
                time.sleep(inter_call_sleep_s)

    return all_results


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def build_report(
    results: List[TaskResult],
    output_dir: str,
    n_seeds: int,
    n_tasks: int,
) -> Tuple[str, str]:
    """Write JSON + Markdown reports. Returns the two file paths."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "a_b_results.json")
    md_path = os.path.join(output_dir, "a_b_report.md")

    pairs = [(r.nare_correct, r.vanilla_correct) for r in results]
    n = len(results)
    nare_correct = sum(1 for a, _ in pairs if a)
    vanilla_correct = sum(1 for _, b in pairs if b)
    nare_acc = nare_correct / n if n else 0.0
    vanilla_acc = vanilla_correct / n if n else 0.0
    nare_low, nare_high = wilson_ci(nare_correct, n)
    van_low, van_high = wilson_ci(vanilla_correct, n)
    diff_point, diff_low, diff_high = paired_bootstrap_diff_ci(pairs)

    nare_lat_mean, nare_lat_std = _mean_std([r.nare_latency_s for r in results])
    van_lat_mean, van_lat_std = _mean_std([r.vanilla_latency_s for r in results])

    # NARE-only routing breakdown
    route_counts: Dict[str, int] = {}
    for r in results:
        route_counts[r.nare_route] = route_counts.get(r.nare_route, 0) + 1
    amortized = sum(
        c for k, c in route_counts.items()
        if k in {"FAST", "REFLEX", "REFLEX_PROVISIONAL", "HYBRID"}
    )
    amortized_pct = (100 * amortized / n) if n else 0.0

    # Per-category accuracy
    by_cat: Dict[str, List[Tuple[bool, bool]]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(
            (r.nare_correct, r.vanilla_correct)
        )

    # ---- JSON dump ----
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_tasks": n_tasks,
                "n_seeds": n_seeds,
                "n_total": n,
                "nare": {
                    "accuracy": nare_acc,
                    "ci95": [nare_low, nare_high],
                    "latency_mean_s": nare_lat_mean,
                    "latency_std_s": nare_lat_std,
                    "amortized_pct": amortized_pct,
                    "route_counts": route_counts,
                },
                "vanilla": {
                    "accuracy": vanilla_acc,
                    "ci95": [van_low, van_high],
                    "latency_mean_s": van_lat_mean,
                    "latency_std_s": van_lat_std,
                },
                "diff": {
                    "nare_minus_vanilla": diff_point,
                    "ci95_paired_bootstrap": [diff_low, diff_high],
                },
                "per_category": {
                    cat: {
                        "n": len(p),
                        "nare_acc": sum(1 for a, _ in p if a) / len(p),
                        "vanilla_acc": sum(1 for _, b in p if b) / len(p),
                    }
                    for cat, p in by_cat.items()
                },
                "tasks": [r.to_json() for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---- Markdown report ----
    lines: List[str] = []
    lines.append("# A/B Benchmark — NARE vs Vanilla Gemini\n")
    lines.append(
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · "
        f"{n_tasks} tasks × {n_seeds} seed(s) = {n} runs._\n"
    )

    lines.append("## Headline\n")
    lines.append(
        f"| Arm | Accuracy | 95% CI (Wilson) | Latency mean ± std |"
    )
    lines.append("|---|---|---|---|")
    lines.append(
        f"| **NARE** | {nare_acc * 100:5.1f}% ({nare_correct}/{n}) | "
        f"[{nare_low * 100:.1f}%, {nare_high * 100:.1f}%] | "
        f"{nare_lat_mean:.2f}s ± {nare_lat_std:.2f}s |"
    )
    lines.append(
        f"| Vanilla Gemma-3-27B (CoT) | {vanilla_acc * 100:5.1f}% "
        f"({vanilla_correct}/{n}) | "
        f"[{van_low * 100:.1f}%, {van_high * 100:.1f}%] | "
        f"{van_lat_mean:.2f}s ± {van_lat_std:.2f}s |"
    )
    lines.append("")
    lines.append(
        f"**Δ (NARE − Vanilla):** {diff_point * 100:+.1f} pp · "
        f"95% paired-bootstrap CI [{diff_low * 100:+.1f} pp, "
        f"{diff_high * 100:+.1f} pp]\n"
    )

    sig = "no"
    if diff_low > 0:
        sig = "**yes — NARE significantly better**"
    elif diff_high < 0:
        sig = "**yes — vanilla significantly better**"
    lines.append(f"_Statistically significant at α=0.05?_ {sig}\n")

    lines.append("## NARE routing breakdown\n")
    lines.append("| Route | Count | Share |")
    lines.append("|---|---|---|")
    for route in sorted(route_counts.keys(), key=lambda k: -route_counts[k]):
        c = route_counts[route]
        share = 100 * c / n
        lines.append(f"| {route} | {c} | {share:.1f}% |")
    lines.append("")
    lines.append(
        f"**Amortized (FAST + REFLEX + HYBRID):** "
        f"{amortized}/{n} ({amortized_pct:.1f}%)\n"
    )

    lines.append("## Per-category accuracy\n")
    lines.append("| Category | n | NARE | Vanilla | Δ |")
    lines.append("|---|---|---|---|---|")
    for cat in sorted(by_cat.keys()):
        p = by_cat[cat]
        n_c = len(p)
        a = sum(1 for x, _ in p if x) / n_c
        b = sum(1 for _, x in p if x) / n_c
        lines.append(
            f"| {cat} | {n_c} | {a * 100:.1f}% | {b * 100:.1f}% | "
            f"{(a - b) * 100:+.1f} pp |"
        )
    lines.append("")

    lines.append("## How to read this\n")
    lines.append(
        "- **NARE** uses the 4-tier router with sleep-phase skill "
        "induction and a per-task `oracle_spec`. The same oracle "
        "grades both arms.\n"
        "- **Vanilla** is a single zero-shot CoT call to the same "
        "Gemma-3-27B-IT through the same client. No memory, no critic, "
        "no sleep, no per-task oracle on the model side (the oracle "
        "still grades).\n"
        "- A positive **Δ** with a CI strictly above 0 is the only "
        "honest claim that NARE outperforms the underlying LLM. A "
        "negative Δ means NARE is worse on this dataset — not "
        "necessarily a bug, but a signal to investigate (often the "
        "skill induction ships buggy code that overrides a correct "
        "vanilla answer).\n"
        "- **Amortized %** is NARE-only and only meaningful when the "
        "memory store is warmed (paraphrased queries within the run "
        "or `--persist-dir` reused across runs).\n"
    )

    # ---- Learning curve (only meaningful when paraphrases > 1) -----
    has_paraphrases = any(getattr(r, "paraphrase_idx", 0) > 0 for r in results)
    if has_paraphrases:
        # Bucket results by paraphrase index across all base tasks.
        by_p_idx: Dict[int, List[TaskResult]] = {}
        for r in results:
            by_p_idx.setdefault(r.paraphrase_idx, []).append(r)

        # By-paraphrase aggregate: NARE accuracy / NARE latency / route mix.
        lines.append("## Learning curve (NARE memory warm-up across paraphrases)\n")
        lines.append(
            "Each base task is run with its original query first, then "
            "1+ paraphrases of the same problem against the same NARE "
            "agent (so memory persists between paraphrases). If NARE "
            "is actually learning, the *p1+* rows should show lower "
            "latency and a higher amortized share than *p0*.\n"
        )
        lines.append(
            "| Paraphrase | n | NARE acc | NARE latency mean ± std | "
            "Vanilla latency mean | Amortized | Route mix |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for p_idx in sorted(by_p_idx):
            bucket = by_p_idx[p_idx]
            nb = len(bucket)
            nare_lat_m, nare_lat_s = _mean_std([r.nare_latency_s for r in bucket])
            van_lat_m, _ = _mean_std([r.vanilla_latency_s for r in bucket])
            nare_acc_b = sum(1 for r in bucket if r.nare_correct) / nb
            amort_b = sum(
                1 for r in bucket
                if r.nare_route in {"FAST", "REFLEX", "REFLEX_PROVISIONAL", "HYBRID"}
            )
            route_mix: Dict[str, int] = {}
            for r in bucket:
                route_mix[r.nare_route] = route_mix.get(r.nare_route, 0) + 1
            mix_str = " · ".join(
                f"{k}={v}" for k, v in sorted(route_mix.items(), key=lambda kv: -kv[1])
            )
            label = f"p{p_idx} (original)" if p_idx == 0 else f"p{p_idx}"
            lines.append(
                f"| {label} | {nb} | {nare_acc_b * 100:.1f}% | "
                f"{nare_lat_m:.2f}s ± {nare_lat_s:.2f}s | "
                f"{van_lat_m:.2f}s | "
                f"{amort_b}/{nb} ({100 * amort_b / nb:.0f}%) | {mix_str} |"
            )
        lines.append("")

        # Per-base-task trajectory: shows route progression for each task.
        lines.append("### Per-task trajectory (route + latency by paraphrase)\n")
        lines.append(
            "Reading guide: `SLOW(6.2s) → FAST(0.7s) → FAST(0.7s)` is the "
            "design point — first run pays the SLOW cost, paraphrases "
            "ride the cache. `SLOW → SLOW → SLOW` means NARE never "
            "induced a reusable skill for this task.\n"
        )
        # Group by (base_task_id, seed) so each seed gets its own
        # trajectory row — different seeds use independent NARE agent
        # instances and merging them would produce a meaningless
        # trajectory (e.g. SLOW(seed=0) → SLOW(seed=1) → FAST(seed=0)).
        # Preserve order of first occurrence.
        seeds_seen = sorted({r.seed for r in results})
        show_seed = len(seeds_seen) > 1
        if show_seed:
            lines.append("| Base task | Seed | Category | Trajectory |")
            lines.append("|---|---|---|---|")
        else:
            lines.append("| Base task | Category | Trajectory |")
            lines.append("|---|---|---|")
        order: List[Tuple[str, int]] = []
        by_base: Dict[Tuple[str, int], List[TaskResult]] = {}
        for r in results:
            key = (r.base_task_id, r.seed)
            if key not in by_base:
                order.append(key)
            by_base.setdefault(key, []).append(r)
        for key in order:
            base, seed = key
            traj = sorted(by_base[key], key=lambda r: r.paraphrase_idx)
            cat = traj[0].category
            steps = " → ".join(
                f"{r.nare_route}({r.nare_latency_s:.1f}s)"
                + ("" if r.nare_correct else "✗")
                for r in traj
            )
            if show_seed:
                lines.append(f"| {base} | {seed} | {cat} | {steps} |")
            else:
                lines.append(f"| {base} | {cat} | {steps} |")
        lines.append("")

    lines.append("## Failure cases (first 10)\n")
    failures = [
        r for r in results if not r.nare_correct or not r.vanilla_correct
    ][:10]
    if not failures:
        lines.append("_All tasks correct on both arms._\n")
    else:
        lines.append("| ID | Cat | Expected | NARE | NARE info | Vanilla | Vanilla info |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in failures:
            lines.append(
                f"| {r.task_id} | {r.category} | `{r.expected_solution[:30]}` "
                f"| {'✓' if r.nare_correct else '✗'} `{r.nare_answer[:40]}` "
                f"| {r.nare_oracle_info[:40]} "
                f"| {'✓' if r.vanilla_correct else '✗'} `{r.vanilla_answer[:40]}` "
                f"| {r.vanilla_oracle_info[:40]} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _load_dataset(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return list(BUILTIN_TASKS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Dataset must be a JSON list of task dicts; got {type(data).__name__}"
        )
    for t in data:
        if not all(k in t for k in ("id", "query", "oracle_spec")):
            raise ValueError(
                f"Task is missing required keys 'id'/'query'/'oracle_spec': {t!r}"
            )
    return data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark — NARE vs vanilla Gemini.",
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Path to a JSON list of task dicts "
            "(see module docstring). Defaults to a built-in 15-task set."
        ),
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help=(
            "Truncate the dataset to this many tasks (default: all)."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help=(
            "Number of independent seeds. Each seed runs the full task "
            "set with a fresh memory store (unless --persist-dir is "
            "set). Default: 1."
        ),
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help=(
            "Reuse this memory_store across seeds (warm cache). "
            "Default: fresh tmp dir per seed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Where to write a_b_results.json and a_b_report.md.",
    )
    parser.add_argument(
        "--paraphrases",
        type=int,
        default=1,
        help=(
            "Number of paraphrased queries to run per base task. "
            "1 (default) preserves the legacy one-query-per-task "
            "behaviour. Values >1 expand each task into the original "
            "query plus up to N-1 paraphrases (from the task's "
            "'paraphrases' field), run sequentially against the same "
            "NARE agent. This is the regime where NARE's memory-warm "
            "amortization can actually manifest."
        ),
    )
    parser.add_argument(
        "--inter-call-sleep",
        type=float,
        default=0.0,
        help=(
            "Optional pause between LLM calls (seconds). Useful when "
            "you are hitting Gemini free-tier rate limits."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-task console output.",
    )
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.getenv("GEMINI_API_KEY"):
        print(
            "ERROR: GEMINI_API_KEY is not set. "
            "Export it before running this benchmark.",
            file=sys.stderr,
        )
        return 2

    dataset = _load_dataset(args.dataset)
    if args.num_tasks is not None:
        dataset = dataset[: args.num_tasks]
    if not dataset:
        print("ERROR: empty dataset.", file=sys.stderr)
        return 2

    if args.paraphrases < 1:
        print("ERROR: --paraphrases must be >= 1.", file=sys.stderr)
        return 2

    seeds = list(range(args.num_seeds))
    # Pre-expand once so the run-count display, the report header, and
    # run_a_b all agree on the same count. _expand_paraphrases is a
    # pure function so calling it again inside run_a_b is harmless.
    expanded_per_seed = _expand_paraphrases(dataset, args.paraphrases)
    runs_per_seed = len(expanded_per_seed)
    total_runs = runs_per_seed * len(seeds)
    print(
        f"\nA/B benchmark - {len(dataset)} base task(s) x "
        f"{args.paraphrases} paraphrase(s) x {len(seeds)} seed(s) = "
        f"{total_runs} runs."
    )
    if args.persist_dir:
        print(f"  Warm memory_store: {args.persist_dir}")
    else:
        print("  Memory: fresh tmp dir per seed (cold start each seed)")
    if args.paraphrases > 1:
        print(
            f"  Paraphrase mode: {args.paraphrases} variant(s) per task "
            f"(memory persists across paraphrases)."
        )
    print()

    t0 = time.time()
    results = run_a_b(
        dataset,
        seeds,
        args.persist_dir,
        inter_call_sleep_s=args.inter_call_sleep,
        paraphrases=args.paraphrases,
    )
    elapsed = time.time() - t0

    json_path, md_path = build_report(
        results,
        args.output_dir,
        n_seeds=len(seeds),
        # n_tasks is the per-seed run count (post-paraphrase-expansion)
        # so the report header arithmetic (n_tasks × n_seeds = n_total)
        # actually holds.
        n_tasks=runs_per_seed,
    )
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min).")
    print(f"  JSON:    {json_path}")
    print(f"  Report:  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
