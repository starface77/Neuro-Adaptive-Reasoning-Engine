# NARE — Known Limitations

This document is intentionally honest. It exists because the README and earlier theoretical drafts described NARE in language that, taken literally, oversells what the code actually does. We list the gaps here so users, reviewers, and future contributors know exactly what works, what is heuristic, and what is open.

If you find a limitation we missed, please open an issue.

---

## 1. No standard benchmark results

The `benchmarks/` directory contains **scripted demonstrations**, not benchmarks:

- 5–7 hand-crafted tasks per file, all from one family (numeric sequences, Kadane-style subarrays, email extraction).
- Sequences are arranged so that SLOW → SLEEP → REFLEX → FAST happens in a predictable order.
- No held-out test set, no baselines (zero-shot / CoT / Reflexion / EXPEL / Voyager), no multiple seeds, no confidence intervals.

To make any quantitative claim about NARE's reasoning quality or efficiency, the system must be run on at least one widely-used benchmark with proper baselines. Suggested:

| Domain | Benchmark | Baselines |
| --- | --- | --- |
| Code | HumanEval+ / MBPP+ | zero-shot, CoT, Reflexion |
| Math | MATH, GSM8K | zero-shot, CoT, Self-Consistency, ToT |
| Multi-step reasoning | BIG-Bench Hard, ARC-Challenge | zero-shot, CoT, ReAct |
| Agentic | AlfWorld, WebArena | ReAct, Reflexion, EXPEL, Voyager |

Until that exists, do not cite NARE's "speedup" numbers as scientific claims. The previously reported "8,500× speedup" was measured against a SLOW path that intentionally inserted `time.sleep(15)` between API calls to respect Gemini free-tier rate limits — that comparison is not meaningful.

## 2. Theory–implementation mapping is metaphorical, not formal

The earlier README and the long-form theory document used vocabulary from:

- **Free Energy Principle / Active Inference (Friston)** — variational free energy, expected free energy, Bayesian model reduction.
- **Topological data analysis** — "topological trinitarian transformation", search/closure/navigation phases.
- **Complexity theory** — Savitch's theorem, NPSPACE→PSPACE.

The code does **not** instantiate any of these objects:

- There is no posterior `q(s)`, no prior `p(s)`, no KL divergence, no ELBO. `NeuralMemory.compute_surprise` returns `np.std(output)` or Huber loss; neither is `−log p(o)`.
- There is no persistent-homology computation, no Betti numbers, no Mapper, no filtration.
- The cache hierarchy does not change the formal complexity class of any problem; it is a memoization strategy.

These framings should be read as **inspirations**, not theorems. If you need the formal versions, cite the original papers.

## 3. Validation signal: self-judging weight removed (was leaky)

**Status:** mitigated as of the oracle-integration PR. Previously, `extract_heuristic_rule` (in `nare/llm.py`) generated skills with one LLM call, generated "stress test" queries-and-labels with the same LLM, then counted those labels for **30% of `overall`**. When the skill model and the label model are the same family, that 30% was self-referential: systematic errors were never caught.

Current behaviour:

* `_validate_skill` now accepts an `oracle` argument (and uses an episode's `oracle_spec` first, then the caller-provided oracle, then a documented heuristic fallback).
* `overall` weights live in `nare.config.SkillValidationConfig`. The new defaults are:
  * `w_trigger = 0.35` &nbsp;(real, labelled originals)
  * `w_execute = 0.55` &nbsp;(oracle-judged against verified solutions)
  * `w_negative_trap = 0.10` &nbsp;(real signal: must not trigger on adversarial off-distribution queries)
  * `w_positive_stress = 0.0` &nbsp;(POSITIVE LLM-judged stress is reported as `positive_no_crash_rate` for diagnostics but excluded from overall by default).
* Hard gates: a skill that fails on its own training originals (`trigger < 0.50` or `execute < 0.40`) is capped at `overall = 0.50` regardless of stress luck.
* `NAREProductionAgent(oracle=...)` propagates the oracle into the sleep / REM phases.
* `nare/oracle.py` ships `numeric_set_oracle`, `string_contains_oracle`, `python_assert_oracle`, `heuristic_overlap_oracle` (the documented fallback), and `build_oracle_from_spec` for JSON-serializable per-episode specs.

What is **still** weak: when neither an episode `oracle_spec` nor a global oracle is supplied, `_validate_skill` falls back to `heuristic_overlap_oracle`, which is a string/numeric-overlap check on the stored solution. This is a heuristic, not ground truth — it is just no longer the *only* option. If you want a real benchmark-grade signal, supply a real oracle (or per-episode `oracle_spec`).

## 4. Sandbox is best-effort, not isolation-grade

`nare/sandbox.py` AST-validates generated code against a whitelist of imports / builtins / attribute names, then `exec`s in a restricted globals dict. The validator now blocks all known historical Python sandbox escapes (see `tests/test_sandbox.py`), but **in-process Python sandboxes are fundamentally porous**. For untrusted code in production, run the skill in a separate process with seccomp / firejail / gVisor / pyodide / Wasm. A subprocess-based sandbox is on the roadmap.

What is fixed in this revision (vs. earlier drafts):

- `__import__` removed from the builtins whitelist.
- `eval`, `exec`, `compile`, `open`, `globals`, `locals`, `getattr`, `setattr`, `delattr`, `dir`, `input` blocked as bare-name calls.
- Dunder-attribute access (`__class__`, `__bases__`, `__subclasses__`, `__globals__`, `__code__`, `__closure__`, `__dict__`, `__getattribute__`, etc.) blocked.
- `global` / `nonlocal` statements rejected.
- The dual `exec()` path in `agent.py:501` that previously bypassed the validator is gone — every skill call now goes through `sandbox.safe_call_trigger` / `safe_execute`.

## 5. "Neural memory" is currently advisory only

`nare/neural_memory.py` is a 2-layer NumPy MLP trained online during the sleep phase. Its output is now logged as an auxiliary novelty signal in `solve()`, but it does **not** influence routing decisions, retrieval, or generation. The "Titans / MIRAS-inspired" tag in some comments is aspirational; the original Titans architecture (Behrouz et al., 2024) integrates gradient-based meta-learning *inside* the attention mechanism, which is not what we do here.

Future work: validate whether the novelty score correlates with task hardness on a held-out set; if it does, gate `τ_fast` or scale the SLOW-path candidate budget by it.

## 6. "Meta-abduction" is keyword + boolean-feature clustering

`nare/meta_abduction.py` clusters skills by Jaccard similarity over nine boolean features (`has_regex`, `has_loop`, `has_math`, ...) plus a domain keyword classifier, and produces a textual "meta-rule". This is *syntactic*, not structural — two skills that both happen to use a `for` loop will cluster together regardless of whether they implement the same algorithm.

Real structural-isomorphism detection requires anti-unification, e-graph saturation, or learned code embeddings (CodeT5 / CodeBERT / UniXcoder). The current module is a placeholder for that future work.

## 7. Tree-of-Thoughts is best-of-N with pre-scoring, not full ToT

`llm.tree_of_thoughts` generates `breadth` initial thoughts in one call, scores them in one call, and expands the top-`(breadth+1)//2` to full solutions. There is no DFS/BFS with back-tracking; pruned branches stay pruned. With the default `depth=1` (used by the agent), this is functionally `best-of-N` with a learned ranker. Renaming this would be more honest; we have not yet, to keep the public API stable.

## 8. Concurrency

The sleep / REM phase runs on a daemon thread and mutates `MemorySystem` data structures. A reentrant lock now wraps every read and write path on `MemorySystem`. This closes the visible race conditions on FAISS index rebuilds, but does not make the system distributed-safe (multiple processes sharing the same `memory_store/` directory will still race on the JSON / `.faiss` files).

## 9. Magic numbers are now named

All previously-scattered thresholds (`tau_fast`, sleep cluster density, dedup thresholds, RL learning rate, blend weights, Elo K-factor, maturity streak, shadow-check window, etc.) are collected in `nare/config.py`. **Their default values are starting points, not optimized values.** No ablation study has been run. Treat the defaults as "reasonable enough to demo", not "tuned".

## 10. No multimodal / no agentic loop / no tool use

NARE today is a single-turn `query: str → solution: str` pipeline. It does not handle images, tool calls, multi-turn dialogues, or agentic environments. Comparing it to systems like ReAct / Reflexion / Voyager / EXPEL on agentic benchmarks would require adding those interfaces.
