import os
import shutil
from nare.agent import NAREProductionAgent
from nare.memory import MemorySystem
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_hardcore_benchmark():
    print("--- NARE HARDCORE REASONING BENCHMARK (Gemma-3-27B) ---")
    agent = NAREProductionAgent()
    
    # Полная очистка памяти перед тестом
    if os.path.exists("memory_store"):
        shutil.rmtree("memory_store")
    agent.memory = MemorySystem()
    
    tasks = [
        # 1. SLOW: Модель должна вывести формулу n^2 + n + 17
        "Find the next two numbers in the sequence: 19, 23, 29, 37, 47... and derive the general formula T(n).",
        
        # 2. SLOW -> SLEEP: Модель должна закрепить метод разностей (Quadratic induction)
        "Find the formula and the next term for: 5, 14, 27, 44, 65... (Formula is 2n^2 + 3n).",
        
        # 3. HYBRID (Delta): Похожая структура, но с условием "аномалии"
        "Find the formula for 6, 15, 28, 45, 66... but assume the first term '6' is a measurement error and should have been '5'. How does this change the general rule?",
        
        # 4. REFLEX: Прямое применение навыка на огромном числе (где CoT обычно ошибается)
        "Using the quadratic induction rule you've learned, calculate the 500th term of the sequence starting with 7, 17, 31, 49, 71...",
        
        # 5. FAST: Повтор сложной задачи
        "Find the next two numbers in the sequence: 19, 23, 29, 37, 47... and derive the general formula T(n)."
    ]

    for i, task in enumerate(tasks):
        print(f"\n{'='*80}")
        print(f"TASK {i+1}: {task}")
        print('='*80)
        
        result = agent.solve(task)
        
        print(f"\n>>> RESULT:")
        print(f"    ROUTE: {result['route_decision']}")
        
        if result['route_decision'] != "FAST":
            candidates = result.get('generated_candidates', [])
            if candidates:
                cand = candidates[0]
                print(f"    TRACE: {cand.get('reasoning', 'N/A')[:500]}...")
            print(f"    FINAL ANSWER: {result['final_answer'][:200]}...")
        
        print("\n    LOGS:")
        for log in result['memory_update_log']:
            print(f"      {log}")

if __name__ == "__main__":
    run_hardcore_benchmark()
