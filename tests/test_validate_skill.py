"""Tests for the oracle-aware skill validator (`nare.llm._validate_skill`).

These tests exercise:

  * The execute path is judged through the supplied Oracle (not through
    the legacy string/numeric overlap), so a candidate that produces
    the right answer in a different surface form is now accepted.
  * Per-episode ``oracle_spec`` overrides the global oracle.
  * NEGATIVE stress traps still gate ``overall``.
  * POSITIVE LLM-judged stress tests do NOT bias ``overall`` under the
    default config (their numeric labels are self-referential).
  * Hard gates: a skill that fails on its own training originals is
    capped regardless of stress luck.

The tests do not call any LLM; they construct skill code, episodes,
and stress tests directly and call ``_validate_skill`` synchronously.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nare.config import DEFAULT_CONFIG, SkillValidationConfig
from nare.llm import _validate_skill
from nare.oracle import (
    build_oracle_from_spec,
    numeric_set_oracle,
    python_assert_oracle,
)


# We deliberately avoid regex backslash-soup inside this Python string
# literal. Trigger fires on any query that contains a '+' AND at least
# two digit groups; that's enough for the test fixtures below.
GOOD_SKILL = """
def trigger(query: str) -> bool:
    if '+' not in query:
        return False
    digits = []
    cur = ''
    for c in query:
        if c.isdigit():
            cur += c
        elif cur:
            digits.append(cur)
            cur = ''
    if cur:
        digits.append(cur)
    return len(digits) >= 2

def execute(query: str) -> str:
    nums = []
    cur = ''
    for c in query + ' ':
        if c.isdigit():
            cur += c
        elif cur:
            nums.append(int(cur))
            cur = ''
    if len(nums) < 2:
        return 'Error: not enough numbers'
    return f'sum is {sum(nums)}'
"""


BAD_SKILL_WRONG_ANSWER = """
def trigger(query: str) -> bool:
    return '+' in query and any(c.isdigit() for c in query)

def execute(query: str) -> str:
    return '42'  # always wrong
"""


def _episode(query: str, solution: str, **extra) -> dict:
    """Build a minimal 'original' episode (has an embedding marker)."""
    ep = {"query": query, "solution": solution, "embedding": [0.0]}
    ep.update(extra)
    return ep


def test_oracle_judges_execute_in_alternate_form():
    """Oracle accepts a structurally-correct answer even when the surface
    string differs from the stored solution. The legacy heuristic
    overlap would fail on this because '7' and 'the answer is 7' don't
    share enough numeric overlap with 'sum is 7' for some cases.
    """

    episodes = [
        _episode("2 + 5", "7"),
        _episode("3 + 4", "7"),
    ]

    # Oracle: numeric set match against the unique number in the query.
    def oracle(q: str, ans: str):
        import re as _re

        nums = [int(x) for x in _re.findall(r"\d+", q)]
        if not nums:
            return False, "no nums in query"
        expected = sum(nums)
        ans_nums = {int(x) for x in _re.findall(r"-?\d+", ans)}
        if expected in ans_nums:
            return True, "oracle match"
        return False, f"expected {expected}"

    scores, _ = _validate_skill(GOOD_SKILL, episodes, oracle=oracle)
    assert scores["execute_accuracy"] == pytest.approx(1.0)
    # No stress tests provided, so negative_trap defaults to 1.0
    # and overall should be high.
    assert scores["overall"] >= 0.95


def test_oracle_rejects_wrong_answer():
    episodes = [_episode("2 + 5", "7"), _episode("3 + 4", "7")]
    oracle = numeric_set_oracle([7])
    scores, _ = _validate_skill(BAD_SKILL_WRONG_ANSWER, episodes, oracle=oracle)
    # Wrong answer "42" rejected by oracle on every original.
    assert scores["execute_accuracy"] == 0.0
    # Hard gate: execute < minimum_execute_accuracy => overall capped.
    assert scores["overall"] <= 0.50


def test_per_episode_oracle_spec_overrides_global():
    """When an episode carries oracle_spec, it must be used instead of
    the global oracle argument.
    """

    # Global oracle says everything is correct; per-episode oracle
    # disagrees. The per-episode one must win.
    permissive_oracle = lambda q, a: (True, "permissive")  # noqa: E731

    episodes = [
        _episode(
            "2 + 5",
            "7",
            oracle_spec={"type": "numeric_set", "expected": [999]},
        ),
        _episode(
            "3 + 4",
            "7",
            oracle_spec={"type": "numeric_set", "expected": [999]},
        ),
    ]

    scores, _ = _validate_skill(GOOD_SKILL, episodes, oracle=permissive_oracle)
    # Skill outputs "sum is 7"; per-episode oracle wants 999. So execute
    # should be 0, overall capped by hard gate.
    assert scores["execute_accuracy"] == 0.0


def test_negative_traps_gate_overall():
    """A skill that fires on NEGATIVE traps loses negative_trap_accuracy
    weight in overall.
    """

    episodes = [
        _episode("2 + 5", "7"),
    ]
    # NEGATIVE trap: not arithmetic; trigger() should NOT fire. Our
    # GOOD_SKILL trigger checks for r"\d+\s*\+\s*\d+", so a query with
    # no '+' should not trigger.
    stress = [{"type": "NEGATIVE", "query": "what is the capital of France?", "solution": "IGNORE"}]
    scores, _ = _validate_skill(
        GOOD_SKILL, episodes + stress, oracle=numeric_set_oracle([7])
    )
    assert scores["negative_trap_accuracy"] == 1.0

    # Now a TRAP that DOES match the regex (so the skill incorrectly
    # fires). negative_trap_accuracy must drop.
    bad_stress = [{"type": "NEGATIVE", "query": "10 + 20 in different category", "solution": "IGNORE"}]
    scores, _ = _validate_skill(
        GOOD_SKILL, episodes + bad_stress, oracle=numeric_set_oracle([7])
    )
    assert scores["negative_trap_accuracy"] == 0.0


def test_positive_stress_no_longer_biases_overall_by_default():
    """POSITIVE stress tests with LLM-generated labels must not move
    ``overall`` under the default config.
    """

    episodes = [_episode("2 + 5", "7")]
    # Two POSITIVE stress tests with completely fabricated solutions.
    # Under the OLD code, the skill returning "sum is 12" would have
    # passed (no-crash), bumping stress_accuracy and thus overall by
    # 0.30 * 0.5. Under the new defaults (w_positive_stress=0), this
    # signal is ignored.
    pos_stress_pass = [
        {"type": "POSITIVE", "query": "5 + 7", "solution": "12"},
        {"type": "POSITIVE", "query": "6 + 6", "solution": "12"},
    ]
    pos_stress_fail = [
        {"type": "POSITIVE", "query": "5 + 7 fail", "solution": "12"},
        {"type": "POSITIVE", "query": "6 + 6 fail", "solution": "12"},
    ]

    base_scores, _ = _validate_skill(
        GOOD_SKILL, episodes, oracle=numeric_set_oracle([7])
    )
    pass_scores, _ = _validate_skill(
        GOOD_SKILL, episodes + pos_stress_pass, oracle=numeric_set_oracle([7])
    )
    fail_scores, _ = _validate_skill(
        GOOD_SKILL, episodes + pos_stress_fail, oracle=numeric_set_oracle([7])
    )

    assert pass_scores["overall"] == pytest.approx(base_scores["overall"])
    assert fail_scores["overall"] == pytest.approx(base_scores["overall"])
    # The advisory metric is still surfaced for diagnostics.
    assert "positive_no_crash_rate" in pass_scores


def test_include_positive_stress_requires_oracle_to_pass():
    """When ``include_positive_stress=True``, POSITIVE stress is only
    counted toward overall when the oracle agrees \u2014 not when only the
    model agrees with itself.
    """

    custom = SkillValidationConfig(
        w_trigger=0.30,
        w_execute=0.40,
        w_negative_trap=0.15,
        w_positive_stress=0.15,
        include_positive_stress=True,
    )
    config = replace(DEFAULT_CONFIG, skill_validation=custom)

    # Originals carry their own oracle_spec, so the *global* oracle
    # below is only consulted for the POSITIVE stress test (which is
    # what include_positive_stress controls).
    episodes = [
        _episode(
            "2 + 5",
            "7",
            oracle_spec={"type": "numeric_set", "expected": [7]},
        ),
    ]
    pos_stress = [
        {"type": "POSITIVE", "query": "10 + 5", "solution": "ignored"},
    ]

    # Permissive global oracle accepts the stress-test output.
    accepting_oracle = lambda q, a: (True, "ok")  # noqa: E731
    scores_ok, _ = _validate_skill(
        GOOD_SKILL,
        episodes + pos_stress,
        oracle=accepting_oracle,
        config=config,
    )
    assert scores_ok["overall"] >= 0.85

    # Rejecting oracle ONLY for the stress test must drag overall down,
    # while leaving execute_accuracy on the originals (which use their
    # own oracle_spec) intact.
    rejecting_oracle = lambda q, a: (False, "no")  # noqa: E731
    scores_bad, _ = _validate_skill(
        GOOD_SKILL,
        episodes + pos_stress,
        oracle=rejecting_oracle,
        config=config,
    )
    assert scores_bad["execute_accuracy"] == pytest.approx(1.0)
    assert scores_bad["overall"] < scores_ok["overall"]


def test_hard_gate_when_trigger_below_minimum():
    """A skill that does not even trigger on its training originals is
    capped at 0.50, regardless of negative-trap luck."""

    skill_never_triggers = """
def trigger(query: str) -> bool:
    return False

def execute(query: str) -> str:
    return ""
"""

    episodes = [_episode("2 + 5", "7"), _episode("3 + 4", "7")]
    stress = [
        {"type": "NEGATIVE", "query": "off topic 1", "solution": "IGNORE"},
        {"type": "NEGATIVE", "query": "off topic 2", "solution": "IGNORE"},
    ]
    scores, _ = _validate_skill(
        skill_never_triggers,
        episodes + stress,
        oracle=numeric_set_oracle([7]),
    )
    assert scores["trigger_accuracy"] == 0.0
    # Negative traps look great (skill never fires) but hard gate fires.
    assert scores["negative_trap_accuracy"] == 1.0
    assert scores["overall"] <= 0.50


def test_build_oracle_from_spec_python_assert():
    spec = {
        "type": "python_assert",
        "code": "assert int(answer) == 42",
    }
    oracle = build_oracle_from_spec(spec)
    ok, _ = oracle("anything", "42")
    assert ok is True
    ok, _ = oracle("anything", "41")
    assert ok is False


def test_build_oracle_from_spec_unknown_type_raises():
    with pytest.raises(ValueError):
        build_oracle_from_spec({"type": "definitely_not_a_real_type"})
