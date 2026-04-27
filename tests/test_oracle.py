from nare.oracle import (
    numeric_set_oracle,
    python_assert_oracle,
    string_contains_oracle,
)


def test_numeric_set_oracle_pass():
    o = numeric_set_oracle([2, 3, 5])
    ok, _ = o("primes <= 5", "Answer: 2, 3, 5.")
    assert ok


def test_numeric_set_oracle_fail():
    o = numeric_set_oracle([7])
    ok, info = o("seventh prime", "Answer: 11")
    assert not ok
    assert "missing" in info


def test_numeric_set_oracle_floats_within_tol():
    o = numeric_set_oracle([3.14159])
    ok, _ = o("pi", "approximately 3.14159")
    assert ok


def test_string_contains_oracle():
    o = string_contains_oracle(["alice@example.com"])
    assert o("extract email", "found alice@example.com here")[0]
    assert not o("extract email", "no email")[0]


def test_python_assert_oracle_pass():
    o = python_assert_oracle("assert int(answer) == 5")
    assert o("solve", "5")[0]


def test_python_assert_oracle_fail():
    o = python_assert_oracle("assert int(answer) == 5")
    ok, info = o("solve", "6")
    assert not ok
    assert "assertion failed" in info or "crashed" in info
