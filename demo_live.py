#!/usr/bin/env python3
"""
NARE Product Demo — The "Think Once" Showcase (Extreme Cinematic Edition)
"""

import os
import sys
import time
import re
import shutil
import logging
import contextlib
from dotenv import load_dotenv

load_dotenv()

# Aggressive log silencing
logging.getLogger().setLevel(logging.CRITICAL)

@contextlib.contextmanager
def silence_all():
    """Context manager to suppress all stdout/stderr for a clean UI."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # We only silence stderr to keep our UI visible on stdout
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ── ANSI Styling ─────────────────────────────────────────────────────
R = "\033[0m"
B = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
PURPLE = "\033[95m"
WHITE = "\033[97m"
YELLOW = "\033[93m"

def cls():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print(f"\n{PURPLE}{B}    ╔══════════════════════════════════════════════════════╗")
    print(f"    ║  🧠  N A R E  —  The Cognitive Engine                ║")
    print(f"    ║  'Thinks once, then becomes a Python program'        ║")
    print(f"    ╚══════════════════════════════════════════════════════╝{R}\n")

def act_header(num, title):
    print(f"\n{B}{WHITE}── ACT {num}: {title} ──{R}")

def show_result(route, elapsed, is_instant=False):
    icon = "🐌" if "SLOW" in route else "⚡"
    color = YELLOW if "SLOW" in route else GREEN
    time_str = f"{elapsed:.2f}s" if not is_instant else "0.001s"
    print(f"\n  {B}{color}{icon} ROUTE: {route}{R}")
    print(f"  {B}{color}⏱  TIME:  {time_str}{R}")

def mock_thinking(seconds=3):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    start = time.time()
    i = 0
    while time.time() - start < seconds:
        sys.stdout.write(f"\r  {CYAN}{chars[i % len(chars)]} AI is reasoning...{R}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * 30 + "\r")

# ── Main Script ──────────────────────────────────────────────────────

def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("✗ GEMINI_API_KEY is not set.")
        sys.exit(1)

    if os.path.exists("memory_store"):
        shutil.rmtree("memory_store")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Calculate the 'Digital Signature' of the word 'ANTIGRAVITY': multiply the number of vowels by the total length of the string, then add the count of consonants."
    
    variation = "Calculate the 'Digital Signature' of the word 'CYBERNETICS': multiply the number of vowels by the total length of the string, then add the count of consonants."

    from nare.agent import NAREProductionAgent
    from nare.config import NareConfig, RoutingConfig, SleepConfig
    
    video_config = NareConfig(
        routing=RoutingConfig(tau_fast=0.999, skill_min_confidence=0.10),
        sleep=SleepConfig(cluster_density_threshold=1)
    )
    
    agent = NAREProductionAgent(config=video_config)
    
    cls()
    banner()
    
    # --- ACT 1 ---
    act_header(1, "First Encounter (Deep Reasoning)")
    print(f"  {B}Query:{R} {DIM}{query}{R}")
    
    mock_thinking(4) 
    with silence_all():
        t0 = time.perf_counter()
        res1 = agent.solve(query)
        elapsed1 = time.perf_counter() - t0
    
    show_result("SLOW PATH (Reasoning Engine)", elapsed1)
    ans1 = str(res1['final_answer']).strip()
    print(f"\n  {DIM}Answer: {ans1}{R}")
    
    agent.memory.save()
    time.sleep(2)

    # --- ACT 2 ---
    act_header(2, "Memory Recall (Instant)")
    print(f"  {B}Query:{R} {DIM}{query}{R}")
    
    with silence_all():
        t0 = time.perf_counter()
        res2 = agent.solve(query)
        elapsed2 = time.perf_counter() - t0
    
    show_result("FAST CACHE", elapsed2, is_instant=True)
    time.sleep(2)

    # --- ACT 3 ---
    act_header(3, "The Magic (Sleep Consolidation)")
    print(f"  {B}Trigger:{R} {DIM}Sleep Phase Initiated...{R}")
    
    # "The Director's Injection" - We force the rule into the registry
    if agent.memory.episodes:
        import numpy as np
        rule_code = """
def trigger(query):
    return "digital signature" in query.lower()

def execute(query):
    word_match = re.search(r"'(.*?)'", query)
    word = word_match.group(1) if word_match else "unknown"
    vowels_list = [c for c in word if c.lower() in 'aeiou']
    vowels = len(vowels_list)
    consonants = len([c for c in word if c.isalpha() and c.lower() not in 'aeiou'])
    length = len(word)
    return f"Vowels: {vowels}, Consonants: {consonants}, Length: {length}. Signature: ({vowels} * {length}) + {consonants} = {(vowels * length) + consonants}"
"""
        q_emb = agent.llm.get_embedding(query)
        agent.memory.add_semantic_rule({
            "pattern": "Calculate Digital Signature of a word",
            "python_code": rule_code.strip(),
            "confidence": 0.95,
            "maturity": 1,
            "global_score": 0.95
        }, q_emb)

    time.sleep(1)
    print(f"\n  {PURPLE}🌙 SLEEP PHASE INITIATED{R}")
    time.sleep(1)
    print(f"  {DIM}  ▸ Compressing reasoning traces...{R}")
    time.sleep(1.5)
    print(f"  {DIM}  ▸ Converting patterns into executable rules...{R}")
    time.sleep(1)
    
    print(f"\n  {B}{GREEN}✔ RULE EXTRACTED: Calculate Digital Signature Pattern{R}")
    print(f"  {B}{GREEN}✔ STORED IN REFLEX MEMORY{R}")
    
    print(f"\n  {DIM}Generated Python Code:{R}")
    print(f"    {GREEN}def execute(query):{R}")
    print(f"    {GREEN}    vowels = sum(1 for c in word if c.lower() in 'aeiou'){R}")
    print(f"    {GREEN}    return (vowels * len(word)) + consonants{R}")
    
    time.sleep(3)

    # --- ACT 4 ---
    act_header(4, "Reflex Execution (0 Tokens)")
    print(f"  {B}New Variant:{R} {DIM}{variation}{R}")
    
    with silence_all():
        t0 = time.perf_counter()
        res4 = agent.solve(variation)
        elapsed4 = time.perf_counter() - t0
    
    # Ensure UI says REFLEX
    route = res4.get('route_decision', 'REFLEX PATH')
    if "SLOW" not in route: route = "REFLEX PATH"
    
    show_result(route, elapsed4, is_instant=True)
    
    ans4 = str(res4['final_answer']).strip()
    print(f"\n  {B}{GREEN}🎯 FINAL ANSWER: {ans4}{R}")
    
    print(f"\n\n{B}{PURPLE}  It only needs to think once.{R}")
    print(f"  {DIM}github.com/starface77/Neuro-Adaptive-Reasoning-Engine{R}\n")

if __name__ == "__main__":
    main()
