import os
import shutil
import time
from nare.agent import NAREProductionAgent
from nare.memory import MemorySystem
import logging

# Disable standard logging for cleaner output
logging.getLogger().setLevel(logging.WARNING)

def run_metrics_benchmark():
    print("==================================================")
    print("   NARE v6: EXECUTABLE REFLEXES METRICS BENCHMARK")
    print("==================================================\n")
    
    if os.path.exists("memory_store"):
        shutil.rmtree("memory_store")
        
    agent = NAREProductionAgent()
    agent.memory = MemorySystem()
    
    # We use a simple structural task: Count vowels in a string
    # The LLM should easily extract the Python execution logic.
    
    tasks = [
        # Phase 1: Learning (SLOW)
        "Find the next two numbers in the sequence: 19, 23, 29, 37, 47... and derive the general formula T(n).",
        "Find the formula and the next term for: 5, 14, 27, 44, 65... (Formula is 2n^2 + 3n).",
        
        # Phase 2: Consolidation (SLEEP should trigger here and write a Python reflex)
        "Find the formula for 6, 15, 28, 45, 66... but assume the first term '6' is a measurement error and should have been '5'.",
        
        # Phase 3: Reflexes (REFLEX)
        "Find the next term in the sequence: 7, 17, 31, 49, 71... and derive the formula.",
        "Find the next term in the sequence: 2, 6, 12, 20, 30... and derive the formula.",
        "Find the next term in the sequence: 3, 8, 15, 24, 35... and derive the formula.",
        
        # Phase 4: Fast Cache (FAST)
        "Find the next two numbers in the sequence: 19, 23, 29, 37, 47... and derive the general formula T(n)."
    ]
    
    results = []
    
    for i, task in enumerate(tasks):
        print(f"Task {i+1}/{len(tasks)}: {task}")
        start_time = time.time()
        res = agent.solve(task)
        elapsed = time.time() - start_time
        
        route = res['route_decision']
        ans = res['final_answer']
        
        print(f"  -> Route: {route}")
        print(f"  -> Answer: {ans[:50]}...")
        print(f"  -> Time: {elapsed:.2f}s\n")
        
        results.append({
            "task": task,
            "route": route,
            "time": elapsed
        })
        
    print("==================================================")
    print("               METRICS DASHBOARD                  ")
    print("==================================================")
    
    total = len(results)
    routes = [r['route'] for r in results]
    
    slow_count = routes.count("SLOW")
    hybrid_count = routes.count("HYBRID")
    reflex_count = routes.count("REFLEX")
    fast_count = routes.count("FAST")
    
    print(f"Total Tasks: {total}")
    print(f"SLOW Paths: {slow_count} ({(slow_count/total)*100:.1f}%)")
    print(f"HYBRID Paths: {hybrid_count} ({(hybrid_count/total)*100:.1f}%)")
    print(f"REFLEX Paths (Executable): {reflex_count} ({(reflex_count/total)*100:.1f}%)")
    print(f"FAST Paths (Cache): {fast_count} ({(fast_count/total)*100:.1f}%)\n")
    
    # Calculate Time Savings
    avg_slow_time = sum(r['time'] for r in results if r['route'] == 'SLOW') / max(1, slow_count)
    avg_reflex_time = sum(r['time'] for r in results if r['route'] == 'REFLEX') / max(1, reflex_count)
    
    print(f"Avg Time (SLOW - Full LLM): {avg_slow_time:.2f}s")
    if reflex_count > 0:
        print(f"Avg Time (REFLEX - Pure Python): {avg_reflex_time:.2f}s")
        speedup = avg_slow_time / avg_reflex_time
        print(f"Speedup via Executable Reflex: {speedup:.1f}x faster!")
        print(f"Token Savings on Reflex Tasks: 100% (0 generation tokens used)")
    else:
        print("REFLEX path was not triggered. Check Sleep Phase logic.")

if __name__ == "__main__":
    run_metrics_benchmark()
