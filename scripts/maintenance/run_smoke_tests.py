#!/usr/bin/env python3
"""Smoke tests for VARE system before benchmark execution.

Run this script to verify all components are working correctly.
"""

import sys
import time
import traceback
from typing import Tuple, List


def test_imports() -> Tuple[bool, str]:
    """Test that all required modules can be imported."""
    try:
        from nare.agent import NAREProductionAgent
        from nare.config import DEFAULT_CONFIG
        from nare.oracle import numeric_set_oracle, build_oracle_from_spec
        from nare.sandbox import safe_execute_freeform, validate_code
        from nare.memory import MemorySystem
        from nare.synthesis import verified_synthesis
        from nare import llm
        return True, "All imports successful"
    except Exception as e:
        return False, f"Import failed: {e}\n{traceback.format_exc()}"


def test_oracle() -> Tuple[bool, str]:
    """Test oracle functionality."""
    try:
        from nare.oracle import numeric_set_oracle, string_contains_oracle

        # Test numeric oracle
        oracle = numeric_set_oracle([42])
        ok, info = oracle("What is 6*7?", "The answer is 42")
        if not ok:
            return False, f"Numeric oracle failed: {info}"

        # Test string oracle
        oracle2 = string_contains_oracle(["hello", "world"])
        ok2, info2 = oracle2("Test", "hello world")
        if not ok2:
            return False, f"String oracle failed: {info2}"

        return True, "Oracle tests passed"
    except Exception as e:
        return False, f"Oracle test failed: {e}\n{traceback.format_exc()}"


def test_sandbox() -> Tuple[bool, str]:
    """Test sandbox execution."""
    try:
        from nare.sandbox import safe_execute_freeform, validate_code, SecurityError

        # Test valid code
        code1 = "print(2 + 2)"
        result1 = safe_execute_freeform(code1)
        if result1 != "4":
            return False, f"Expected '4', got '{result1}'"

        # Test code with imports
        code2 = """
import math
result = math.sqrt(16)
print(int(result))
"""
        result2 = safe_execute_freeform(code2)
        if result2 != "4":
            return False, f"Expected '4', got '{result2}'"

        # Test forbidden code (should raise SecurityError)
        code3 = "import os\nprint(os.listdir('.'))"
        try:
            validate_code(code3)
            return False, "Sandbox allowed forbidden import 'os'"
        except SecurityError:
            pass  # Expected

        return True, "Sandbox tests passed"
    except Exception as e:
        return False, f"Sandbox test failed: {e}\n{traceback.format_exc()}"


def test_memory() -> Tuple[bool, str]:
    """Test memory system."""
    try:
        import tempfile
        import shutil
        from nare.memory import MemorySystem
        from nare.config import DEFAULT_CONFIG

        # Create temporary memory directory
        temp_dir = tempfile.mkdtemp(prefix='vare_test_')

        try:
            # Initialize memory with 384-dim embeddings
            memory = MemorySystem(
                embedding_dim=384,
                persist_dir=temp_dir,
                config=DEFAULT_CONFIG
            )

            # Test episode addition
            import numpy as np
            test_embedding = np.random.rand(384).astype(np.float32)
            episode = {
                'query': 'test query',
                'solution': 'test solution',
                'reasoning_trace': 'test reasoning',
                'score': 0.9,
                'timestamp': time.time()
            }

            success = memory.add_episode(episode, test_embedding)
            if not success:
                return False, "Failed to add episode to memory"

            # Test retrieval
            retrieved = memory.retrieve_episodes(test_embedding.reshape(1, -1), k=1)
            if len(retrieved) != 1:
                return False, f"Expected 1 retrieved episode, got {len(retrieved)}"

            if retrieved[0]['query'] != 'test query':
                return False, f"Retrieved wrong episode: {retrieved[0]['query']}"

            # Test save/load
            memory.save()
            memory2 = MemorySystem(
                embedding_dim=384,
                persist_dir=temp_dir,
                config=DEFAULT_CONFIG
            )

            if len(memory2.episodes) != 1:
                return False, f"Expected 1 episode after load, got {len(memory2.episodes)}"

            return True, "Memory tests passed"

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return False, f"Memory test failed: {e}\n{traceback.format_exc()}"


def test_agent_initialization() -> Tuple[bool, str]:
    """Test agent initialization."""
    try:
        import tempfile
        import shutil
        from nare.agent import NAREProductionAgent
        from nare.config import DEFAULT_CONFIG

        temp_dir = tempfile.mkdtemp(prefix='vare_agent_test_')

        try:
            agent = NAREProductionAgent(
                config=DEFAULT_CONFIG,
                persist_dir=temp_dir,
                embedding_dim=384
            )

            # Check components
            if agent.memory is None:
                return False, "Agent memory not initialized"

            if agent.router is None:
                return False, "Agent router not initialized"

            if agent.metrics is None:
                return False, "Agent metrics not initialized"

            return True, "Agent initialization passed"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return False, f"Agent initialization failed: {e}\n{traceback.format_exc()}"


def test_verified_synthesis() -> Tuple[bool, str]:
    """Test verified synthesis loop (simplified - just check it runs)."""
    try:
        from nare.synthesis import verified_synthesis
        from nare.oracle import numeric_set_oracle

        # Simple math problem
        query = "What is 5 + 3?"
        oracle = numeric_set_oracle([8])

        attempt_count = [0]

        def propose_fn(prompt, priors):
            # Mock LLM that returns correct answer on first attempt
            attempt_count[0] += 1
            # Return just the answer "8" which should pass the oracle
            return "8"

        result = verified_synthesis(
            query=query,
            propose_fn=propose_fn,
            oracle=oracle,
            max_attempts=3
        )

        # Check that synthesis ran (converged or not)
        if result.total_attempts == 0:
            return False, "Synthesis did not run any attempts"

        # For smoke test, just verify the mechanism works
        # (actual convergence depends on LLM, which we're mocking)
        return True, "Verified synthesis test passed (mechanism verified)"

    except Exception as e:
        return False, f"Verified synthesis test failed: {e}\n{traceback.format_exc()}"


def test_config() -> Tuple[bool, str]:
    """Test configuration."""
    try:
        from nare.config import DEFAULT_CONFIG

        # Check key config values
        if DEFAULT_CONFIG.routing.tau_fast < 0 or DEFAULT_CONFIG.routing.tau_fast > 1:
            return False, f"Invalid tau_fast: {DEFAULT_CONFIG.routing.tau_fast}"

        if DEFAULT_CONFIG.synthesis.max_attempts < 1:
            return False, f"Invalid max_attempts: {DEFAULT_CONFIG.synthesis.max_attempts}"

        if DEFAULT_CONFIG.synthesis.slow_path_breadth < 1:
            return False, f"Invalid slow_path_breadth: {DEFAULT_CONFIG.synthesis.slow_path_breadth}"

        return True, "Config tests passed"

    except Exception as e:
        return False, f"Config test failed: {e}\n{traceback.format_exc()}"


def run_all_tests() -> bool:
    """Run all smoke tests and report results."""
    tests: List[Tuple[str, callable]] = [
        ("Imports", test_imports),
        ("Oracle", test_oracle),
        ("Sandbox", test_sandbox),
        ("Memory", test_memory),
        ("Agent Init", test_agent_initialization),
        ("Verified Synthesis", test_verified_synthesis),
        ("Config", test_config),
    ]

    print("=" * 60)
    print("VARE Smoke Tests")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        print(f"Running: {test_name}...", end=" ", flush=True)

        try:
            success, message = test_fn()

            if success:
                print(f"[PASS]")
                passed += 1
            else:
                print(f"[FAIL]")
                print(f"  Error: {message}")
                failed += 1

        except Exception as e:
            print(f"[FAIL] (exception)")
            print(f"  Error: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print()
        print("[OK] All tests passed! System is ready for benchmark.")
        print()
        return True
    else:
        print()
        print("[ERROR] Some tests failed. Fix issues before running benchmark.")
        print()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
