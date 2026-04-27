"""Sandbox escape attempts. Each must raise SecurityError.

These are NOT exhaustive — the in-process Python sandbox is fundamentally
porous and you should run a real subprocess sandbox in production. But
each test below is a known historical escape that the validator must
catch.
"""

import pytest

from nare.sandbox import (
    SecurityError,
    safe_call_trigger,
    safe_execute,
    safe_load_module,
    validate_code,
)


VALID_SKILL = """
import re

def trigger(query: str) -> bool:
    return bool(re.search(r"\\d+", query))

def execute(query: str) -> str:
    nums = re.findall(r"-?\\d+", query)
    return str(sum(int(n) for n in nums))
"""


def test_valid_skill_loads_and_runs():
    out = safe_execute(VALID_SKILL, "add 2 and 3")
    assert out == "5"


def test_valid_skill_trigger_only():
    triggered, ns = safe_call_trigger(VALID_SKILL, "no nums here")
    assert triggered is False
    assert "execute" in ns


@pytest.mark.parametrize(
    "bad_code,reason",
    [
        # Forbidden imports
        ("import os\ndef trigger(q): return True\ndef execute(q): return ''",
         "import os"),
        ("import sys\ndef trigger(q): return True\ndef execute(q): return ''",
         "import sys"),
        ("from os import system\ndef trigger(q): return True\ndef execute(q): return ''",
         "from os"),
        # eval / exec / compile / open
        ("def trigger(q): return True\ndef execute(q): return eval(q)",
         "eval()"),
        ("def trigger(q): return True\ndef execute(q): return exec(q)",
         "exec()"),
        ("def trigger(q): return True\ndef execute(q): open('/etc/passwd').read()",
         "open()"),
        ("def trigger(q): return True\ndef execute(q): compile(q, '', 'exec')",
         "compile()"),
        # __import__ as bare call
        ("def trigger(q): return True\ndef execute(q): return __import__('os')",
         "__import__()"),
        # Classic ()-class chain escape
        ("def trigger(q): return True\n"
         "def execute(q):\n"
         "    return ().__class__.__bases__[0].__subclasses__()",
         "__class__ chain"),
        # Frame walking
        ("def trigger(q): return True\n"
         "def execute(q):\n"
         "    f = (lambda: None).__globals__\n"
         "    return str(f)",
         "__globals__ chain"),
        # getattr-based escape
        ("def trigger(q): return True\n"
         "def execute(q):\n"
         "    return getattr(q, 'upper')()",
         "getattr()"),
        # globals() / locals()
        ("def trigger(q): return True\ndef execute(q): return str(globals())",
         "globals()"),
        ("def trigger(q): return True\ndef execute(q): return str(locals())",
         "locals()"),
        # global/nonlocal statements
        ("X = 0\n"
         "def trigger(q):\n"
         "    global X\n"
         "    X = 1\n"
         "    return True\n"
         "def execute(q): return str(X)",
         "global statement"),
        # Code object / closure access
        ("def trigger(q): return True\n"
         "def execute(q): return str((lambda: None).__code__)",
         "__code__ chain"),
    ],
)
def test_known_escape_attempts_blocked(bad_code, reason):
    with pytest.raises(SecurityError):
        validate_code(bad_code)


def test_validator_rejects_syntax_errors_as_security():
    with pytest.raises(SecurityError):
        validate_code("def trigger(q):\n    return (")  # incomplete expression


def test_validator_rejects_empty_code():
    with pytest.raises(SecurityError):
        validate_code("")
    with pytest.raises(SecurityError):
        validate_code("   \n  \t  ")


def test_skill_must_define_required_functions():
    only_trigger = "def trigger(q): return True\n"
    with pytest.raises(ValueError):
        safe_execute(only_trigger, "x")


def test_skill_runtime_errors_are_caught():
    bad = (
        "def trigger(q): return True\n"
        "def execute(q): return 1 / 0\n"
    )
    out = safe_execute(bad, "x")
    assert out.startswith("Error:")
    assert "ZeroDivisionError" in out


def test_skill_can_use_re_and_math():
    code = (
        "import re\n"
        "import math\n"
        "def trigger(q): return bool(re.search(r'\\d', q))\n"
        "def execute(q):\n"
        "    n = int(re.findall(r'\\d+', q)[0])\n"
        "    return str(math.factorial(n))\n"
    )
    out = safe_execute(code, "compute 5")
    assert out == "120"
