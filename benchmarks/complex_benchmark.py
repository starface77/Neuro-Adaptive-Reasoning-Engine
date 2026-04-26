import os
from nare.agent import NAREProductionAgent
from nare.memory import MemorySystem
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_complex_benchmark():
    print("Initializing NARE Production Architecture (Gemma-3-27B)...")
    agent = NAREProductionAgent()
    
    # Ensure memory is wiped clean for a fair test
    if os.path.exists("memory_store"):
        import shutil
        shutil.rmtree("memory_store")
    agent.memory = MemorySystem()  # re-init
    
    complex_tasks = [
        # --- PHASE 1: NAIVE REASONING (System 2) ---
        # Task 1: Slow reasoning from scratch.
        "Write a Python function to find the maximum contiguous subarray sum in a list of integers.",
        
        # Task 2: Similar task. Will likely trigger SLOW or HYBRID. 
        # Will trigger SLEEP PHASE because we have 2 similar tasks (density >= 1).
        "Given an array of daily stock prices, find the maximum profit you can achieve with a single buy and sell.",
        
        # --- PHASE 2: DELTA REASONING (Hybrid) ---
        # Task 3: A variation. It's related to subarray sums but asks for something slightly different.
        # Should trigger HYBRID path (Delta Reasoning).
        "Find the maximum contiguous subarray sum, but you are allowed to flip the sign of at most one element.",
        
        # --- PHASE 3: RULE ACTIVATION (System 1 Reflex) ---
        # Task 4: Another maximum subarray problem. Should trigger the Kadane's Algorithm rule extracted in Sleep Phase.
        "Calculate the largest sum of any contiguous sequence in an array of positive and negative numbers.",
        
        # --- PHASE 4: EXACT RETRIEVAL (Fast Path) ---
        # Task 5: Exact string match of Task 2. Should hit FAST path immediately.
        "Given an array of daily stock prices, find the maximum profit you can achieve with a single buy and sell."
    ]

    print("\n" + "="*60)
    print(" STARTING COMPLEX BENCHMARK ")
    print("="*60 + "\n")

    for i, task in enumerate(complex_tasks):
        print(f"\n" + "="*60)
        print(f"Executing Task {i+1}: {task}")
        print("="*60)
        
        result = agent.solve(task)
        
        print(f"\n>>> Task {i+1} Output:")
        print(f"    Route Used: {result['route_decision']}")
        
        if result['route_decision'] == "FAST":
            print(f"    (Pure Memory Retrieval - No generation)")
        else:
            cand = result['generated_candidates'][0] if result['generated_candidates'] else None
            if cand:
                print(f"    [Trace Snippet]: {cand.get('reasoning', 'None')[:200]}...")
                print(f"    [Solution Snippet]: {result['final_answer'][:200]}...\n")
            
        print("    Memory Logs:")
        for log in result['memory_update_log']:
            print(f"      - {log}")

if __name__ == "__main__":
    run_complex_benchmark()
