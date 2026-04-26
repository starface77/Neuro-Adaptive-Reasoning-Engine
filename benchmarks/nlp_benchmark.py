import os
import sys
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nare.agent import NAREProductionAgent
from nare.memory import MemorySystem
import logging

logging.getLogger().setLevel(logging.WARNING)

def run_nlp_benchmark():
    print("==================================================")
    print("   NARE v7: DOMAIN GENERALIZATION BENCHMARK (NLP) ")
    print("==================================================\n")
    
    if os.path.exists("memory_store"):
        shutil.rmtree("memory_store")
        
    agent = NAREProductionAgent()
    agent.memory = MemorySystem()
    
    # We use a string processing task: Extracting emails from messy text.
    # The LLM should extract a Python reflex using regex.
    
    tasks = [
        # Phase 1: Learning (SLOW)
        "Extract the email address from this text: 'Contact us at support@example.com for help.'",
        "Extract the email address from this text: 'My personal email is john.doe123@gmail.com, please write to me.'",
        
        # Phase 2: Consolidation (SLEEP should trigger and write a Python reflex using re.findall)
        "Extract the email address from this text: 'You can reach HR at human.resources@company.net today.'",
        
        # Phase 3: Reflexes (REFLEX)
        "Extract the email address from this text: 'Send your resume to jobs@startup.io before Friday.'",
        "Extract the email address from this text: 'The admin email is admin-test@server.org.'",
        "Extract the email address from this text: 'For billing inquiries, email billing@finance.co.uk.'",
        
        # Phase 4: Fast Cache (FAST)
        "Extract the email address from this text: 'Contact us at support@example.com for help.'"
    ]
    
    results = []
    
    for i, task in enumerate(tasks):
        if getattr(agent, '_is_sleeping', False):
            print("  [Waiting for background Sleep Phase to complete...]")
            while getattr(agent, '_is_sleeping', False):
                time.sleep(1)
                
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

if __name__ == "__main__":
    run_nlp_benchmark()
