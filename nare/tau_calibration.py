"""Replay-driven calibration of FAST-path threshold ``tau_fast``.

Motivation
----------

``tau_fast`` (the cosine-similarity threshold above which the router
short-circuits to a cached solution) was originally a hard-coded
``0.98``. Its only feedback loop is the online ``_calibrate_tau``,
which moves the threshold up or down by a small step based on per-call
reward — useful for drift, but uninformative about the *initial*
threshold relative to the actual cached distribution.

This module computes a held-out, deterministic ROC curve from the
agent's own episode store: for every pair of cached episodes
``(a, b)`` such that ``cos(emb(a), emb(b)) >= tau``, would the FAST
path return a *correct* solution? Correctness is judged by the same
strict ``cached_episode_oracle`` used during skill validation
(normalized-exact OR strict numeric-set, no extra hallucinated
numbers).

The result is a precision/coverage curve over candidate ``tau`` values
plus a recommended threshold that hits a target precision (default
0.95). This calibration is **offline-only** (no LLM calls, no API
key required) and can be re-run any time the episode store changes.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .oracle import cached_episode_oracle


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def _default_oracle_factory(expected_solution: str) -> Callable[[str, str], bool]:
    """Default judge: strict cached-episode oracle on the *target* solution."""
    return cached_episode_oracle(expected_solution)


def calibrate_tau_fast(
    episodes: Sequence[Dict],
    *,
    tau_min: float = 0.90,
    tau_max: float = 0.999,
    tau_step: float = 0.005,
    target_precision: float = 0.95,
    embedding_key: str = "embedding",
    oracle_factory: Callable[[str], Callable[[str, str], bool]] = _default_oracle_factory,
) -> Dict:
    """Compute a precision/coverage curve for ``tau_fast`` over cached episodes.

    For each candidate threshold ``tau``, every ordered pair of distinct
    episodes ``(query=a, donor=b)`` such that
    ``cos(emb(a), emb(b)) >= tau`` is considered a FAST-path activation.
    The activation is correct if applying ``oracle_factory(b['solution'])``
    to ``a['solution']`` returns True (i.e., the donor's solution is also
    valid for the query). The curve reports:

    * ``precision`` — correct activations / total activations
    * ``coverage`` — total activations / total ordered pairs

    Parameters
    ----------
    episodes
        Sequence of episode dicts. Each must contain ``embedding`` (or
        the configured ``embedding_key``) and ``solution`` and ``query``.
    tau_min, tau_max, tau_step
        Sweep range.
    target_precision
        Returned ``recommended_tau`` is the smallest tau on the sweep
        whose precision is at or above this target. ``None`` if no tau
        meets the target.
    embedding_key
        Which embedding to compare. Defaults to ``embedding`` (the raw
        query embedding); pass ``signature_embedding`` to calibrate
        against the abstract-signature space instead.
    oracle_factory
        Callable mapping the donor's solution string to a
        ``(query, candidate) -> bool`` oracle. Defaults to
        :func:`nare.oracle.cached_episode_oracle`.

    Returns
    -------
    dict
        ``{
            "n_episodes": int,
            "n_pairs": int,
            "target_precision": float,
            "recommended_tau": Optional[float],
            "roc": [
                {
                    "tau": float,
                    "n_fast": int,
                    "n_correct": int,
                    "precision": float,
                    "coverage": float,
                },
                ...
            ],
        }``
    """
    if tau_step <= 0:
        raise ValueError("tau_step must be positive")
    if not (0.0 <= tau_min < tau_max <= 1.0):
        raise ValueError("require 0 <= tau_min < tau_max <= 1")

    eligible = [
        ep
        for ep in episodes
        if isinstance(ep, dict)
        and embedding_key in ep
        and isinstance(ep.get("solution"), str)
        and ep["solution"].strip()
    ]
    n = len(eligible)
    if n < 2:
        return {
            "n_episodes": n,
            "n_pairs": 0,
            "target_precision": target_precision,
            "recommended_tau": None,
            "roc": [],
        }

    # Build similarity matrix once.
    raw = np.array([ep[embedding_key] for ep in eligible], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(
            f"embedding column '{embedding_key}' must be a 2D matrix; got shape {raw.shape}"
        )
    vecs = _normalize(raw)
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)  # exclude self-pairs from FAST consideration

    # Pre-compute correctness for every ordered pair (a -> b) where the
    # donor would supply b's solution as the answer to a's query. Doing
    # this once is O(n^2) oracle calls, but episode counts are small
    # (<= a few thousand) and oracle calls are cheap (string ops).
    n_pairs = n * (n - 1)
    correctness = np.zeros((n, n), dtype=bool)
    for i, a in enumerate(eligible):
        a_solution = a.get("solution", "")
        if not isinstance(a_solution, str) or not a_solution.strip():
            continue
        for j, b in enumerate(eligible):
            if i == j:
                continue
            b_solution = b.get("solution", "")
            if not isinstance(b_solution, str) or not b_solution.strip():
                continue
            try:
                judge = oracle_factory(b_solution)
                # Question: would b's solution be accepted as the
                # answer to a? Use a's verified solution as candidate.
                verdict = judge(a.get("query", ""), a_solution)
                # Oracles in :mod:`nare.oracle` return ``(bool, reason)``.
                # Older / custom oracles may return a bare bool — accept
                # either shape.
                if isinstance(verdict, tuple) and verdict:
                    ok = bool(verdict[0])
                else:
                    ok = bool(verdict)
            except Exception as exc:  # noqa: BLE001
                logging.debug(f"[tau_calibration] oracle raised on pair ({i},{j}): {exc}")
                ok = False
            correctness[i, j] = ok

    # Sweep tau.
    n_steps = int(math.floor((tau_max - tau_min) / tau_step)) + 1
    roc: List[Dict] = []
    for k in range(n_steps):
        tau = round(tau_min + k * tau_step, 6)
        fast_mask = sim >= tau
        n_fast = int(fast_mask.sum())
        if n_fast == 0:
            roc.append(
                {
                    "tau": tau,
                    "n_fast": 0,
                    "n_correct": 0,
                    "precision": 1.0,  # vacuous; flagged by coverage=0
                    "coverage": 0.0,
                }
            )
            continue
        n_correct = int((fast_mask & correctness).sum())
        precision = n_correct / n_fast
        coverage = n_fast / n_pairs
        roc.append(
            {
                "tau": tau,
                "n_fast": n_fast,
                "n_correct": n_correct,
                "precision": precision,
                "coverage": coverage,
            }
        )

    # Recommended tau: smallest tau whose precision >= target AND
    # coverage > 0 (vacuous tau=1.0 doesn't count).
    recommended: Optional[float] = None
    for row in roc:
        if row["coverage"] > 0.0 and row["precision"] >= target_precision:
            recommended = row["tau"]
            break

    return {
        "n_episodes": n,
        "n_pairs": n_pairs,
        "target_precision": target_precision,
        "recommended_tau": recommended,
        "roc": roc,
    }


def format_roc_table(report: Dict) -> str:
    """Pretty-print a calibration report as a fixed-width text table."""
    lines = [
        f"# tau_fast calibration  (n={report['n_episodes']} episodes, "
        f"{report['n_pairs']} ordered pairs)",
        f"# target precision = {report['target_precision']:.2f}",
        f"# recommended tau   = {report['recommended_tau']}",
        "",
        f"{'tau':>7}  {'n_fast':>7}  {'n_correct':>9}  {'precision':>9}  {'coverage':>9}",
    ]
    for row in report["roc"]:
        lines.append(
            f"{row['tau']:7.4f}  {row['n_fast']:7d}  {row['n_correct']:9d}  "
            f"{row['precision']:9.4f}  {row['coverage']:9.4f}"
        )
    return "\n".join(lines)


__all__ = [
    "calibrate_tau_fast",
    "format_roc_table",
]
