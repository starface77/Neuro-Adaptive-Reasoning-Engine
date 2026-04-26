import os
import time
from nare.agent import NAREProductionAgent
from dotenv import load_dotenv

def main():
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("[Error] GEMINI_API_KEY environment variable is missing in .env.")
        return

    print("Initializing NARE Production Architecture (Gemini Engine)...")
    agent = NAREProductionAgent()
    
    # Run the Benchmark Tasks
    tasks = [
        # Task 1: Slow Path (Generation & Evaluation)
        "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning.",
        
        # Task 2: Exact Repeat -> Fast Path (Retrieval only)
        "What is the most efficient sorting algorithm for an array that is already 90% sorted? Explain your reasoning.",
        
        # Task 3: Slight Variation -> Hybrid Path
        "Which sorting algorithm should I use if my dataset is mostly sorted, but has a few random elements at the end?",
        
        # We can add more tasks to trigger Sleep Phase (needs 5 similar episodes or >200 total)
        "How to sort an almost sorted list of integers?",
        "Best algorithm for sorting a nearly sorted array in Python?",
        "Sorting an array with only a few inversions?"
    ]
    
    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Executing Task {i+1}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = agent.solve(task)
        elapsed = time.time() - start_time
        
        print("\n[Final Answer]:")
        print(result["final_answer"][:300] + "...\n")
        
        print(f">>> Task {i+1} Metrics:")
        print(f"    Route Used: {result['route_decision']}")
        print(f"    Time Elapsed: {elapsed:.2f} seconds")
        print("    Memory Logs:")
        for log in result['memory_update_log']:
            print(f"      - {log}")

if __name__ == "__main__":
    main()
