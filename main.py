import os
import sys
import time
import argparse
from nare.agent import NAREProductionAgent
from dotenv import load_dotenv


def run_demo(agent: NAREProductionAgent):
    """Run built-in demo tasks showcasing all 4 routing paths."""
    tasks = [
        # Task 1: Slow Path (Generation & Evaluation)
        "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning.",
        # Task 2: Exact Repeat -> Fast Path (Retrieval only)
        "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning.",
        # Task 3: Slight Variation -> Hybrid Path
        "Which sorting algorithm should I use if my dataset is mostly sorted, but has a few random elements at the end?",
        # Additional tasks to trigger Sleep Phase
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


def run_interactive(agent: NAREProductionAgent):
    """Interactive REPL mode for querying NARE."""
    print("NARE Interactive Mode (type 'exit' or 'quit' to stop)\n")

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
        prog="nare",
        description="NARE — Neural Amortized Reasoning Engine (legacy entrypoint)",
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
        choices=["metrics", "complex", "nlp", "hardcore", "basic"],
        default="metrics",
        help="Benchmark suite to run (only used with 'benchmark' mode)",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit",
    )
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("[Error] No API key configured.")
        print("  1. Copy .env.example to .env")
        print("  2. Set ONE of:")
        print("     GEMINI_API_KEY=...      # https://aistudio.google.com/apikey")
        print("     ANTHROPIC_API_KEY=...   # standard or proxy endpoint")
        sys.exit(1)

    print("Initializing NARE...")
    agent = NAREProductionAgent()

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
        if args.benchmark == "metrics":
            from benchmarks.metrics_benchmark import run_metrics_benchmark
            run_metrics_benchmark()
        elif args.benchmark == "complex":
            from benchmarks.complex_benchmark import run_complex_benchmark
            run_complex_benchmark()
        elif args.benchmark == "nlp":
            from benchmarks.nlp_benchmark import run_nlp_benchmark
            run_nlp_benchmark()
        elif args.benchmark == "hardcore":
            from benchmarks.hardcore_benchmark import run_hardcore_benchmark
            run_hardcore_benchmark()
        elif args.benchmark == "basic":
            from benchmarks.benchmark import run_benchmark
            run_benchmark()


if __name__ == "__main__":
    main()
