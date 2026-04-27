import os
import shutil
import time
from collections import defaultdict
from nare.agent import NAREProductionAgent
from nare.memory import MemorySystem

BENCHMARK = [
    ("SLOW", "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning."),
    ("SLOW", "Write a Python function to find the maximum contiguous subarray sum in a list of integers."),
    ("HYBRID", "Which sorting algorithm should I use if my dataset is mostly sorted, but has a few random elements at the end?"),
    ("HYBRID", "Find the maximum contiguous subarray sum, but you are allowed to flip the sign of at most one element."),
    ("FAST", "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning."),
]

def run_benchmark():
    if os.path.exists("memory_store"):
        shutil.rmtree("memory_store")

    agent = NAREProductionAgent()
    agent.memory = MemorySystem()

    results = []
    stats = defaultdict(list)

    for i, (task_type, task) in enumerate(BENCHMARK):

        print(f"\n=== TASK {i+1} [{task_type}] ===")

        start = time.time()
        result = agent.solve(task)
        elapsed = time.time() - start

        route = result["route_decision"]
        memory_hits = len(result.get("memory_update_log", []))

        stats["latency"].append(elapsed)
        stats["route"].append(route)
        stats["memory_hits"].append(memory_hits)

        print(f"Route: {route}")
        print(f"Latency: {elapsed:.3f}s")

        results.append({
            "task_type": task_type,
            "task": task,
            "route": route,
            "time": elapsed,
        })

    print("\n================ BENCHMARK SUMMARY ================")

    print("AVG LATENCY:", sum(stats["latency"]) / len(stats["latency"]))
    print("MEMORY ACTIVITY:", sum(stats["memory_hits"]))
    print("FAST USAGE RATE:", stats["route"].count("FAST") / len(stats["route"]))

    return results

if __name__ == "__main__":
    run_benchmark()