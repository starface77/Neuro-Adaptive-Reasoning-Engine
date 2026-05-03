import os
import sys
import time
import argparse
from nare.agent import VareAgent, NAREProductionAgent
from dotenv import load_dotenv


def run_demo(agent: VareAgent):
    """Run built-in demo tasks showcasing FAST and VERIFIED_RETRY routes."""
    tasks = [
        # Task 1: Verified Retry (first encounter)
        "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning.",
        # Task 2: Exact Repeat -> FAST (cached)
        "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning.",
        # Task 3: Similar -> should benefit from memory
        "Which sorting algorithm should I use if my dataset is mostly sorted, but has a few random elements at the end?",
        # Additional tasks to trigger Library Learning
        "How to sort an almost sorted list of integers?",
        "Best algorithm for sorting a nearly sorted array in Python?",
        "Sorting an array with only a few inversions?"
    ]

    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Executing Task {i+1}/{len(tasks)}...")
        print(f"{'='*60}")

        start_time = time.time()
        result = agent.solve(task)
        elapsed = time.time() - start_time

        print("\n[Final Answer]:")
        answer = result["final_answer"]
        print(answer[:300] + ("..." if len(answer) > 300 else "") + "\n")

        print(f">>> Task {i+1} Metrics:")
        print(f"    Route Used: {result['route_decision']}")
        print(f"    Time Elapsed: {elapsed:.2f} seconds")
        print("    Memory Logs:")
        for log in result['memory_update_log']:
            print(f"      - {log}")


def run_interactive(agent: VareAgent):
    """Interactive REPL mode for querying VARE."""
    print("VARE Interactive Mode (type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        start_time = time.time()
        try:
            result = agent.solve(query)
        except Exception as e:
            print(f"[Error] {e}\n")
            continue
        elapsed = time.time() - start_time

        print(f"\n[Route: {result['route_decision']}] ({elapsed:.2f}s)")
        print(result["final_answer"])
        print()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="vare",
        description="VARE — Verified Amortized Reasoning Engine",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=["demo", "interactive", "benchmark"],
        help="Execution mode: demo (default), interactive, or benchmark",
    )
    parser.add_argument(
        "--benchmark",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark suite to run (only used with 'benchmark' mode)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit",
    )
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("[Error] GEMINI_API_KEY is not set.")
        print("  1. Copy .env.example to .env")
        print("  2. Add your key: GEMINI_API_KEY=your_key_here")
        print("  3. Get a key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("Initializing VARE (Verified Amortized Reasoning Engine)...")
    agent = VareAgent()

    if args.query:
        start_time = time.time()
        result = agent.solve(args.query)
        elapsed = time.time() - start_time
        print(f"\n[Route: {result['route_decision']}] ({elapsed:.2f}s)")
        print(result["final_answer"])
        return

    if args.mode == "demo":
        run_demo(agent)
    elif args.mode == "interactive":
        run_interactive(agent)
    elif args.mode == "benchmark":
        if args.benchmark == "quick":
            from benchmarks.quick_test import main as quick_main
            quick_main()
        elif args.benchmark == "full":
            from benchmarks.full_benchmark import main as full_main
            full_main()


if __name__ == "__main__":
    main()
