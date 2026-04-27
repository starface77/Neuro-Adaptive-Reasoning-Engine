import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from nare.sandbox import ASTValidator, safe_execute
import nare.llm as llm

logging.basicConfig(level=logging.INFO)

episodes = [
    {
        "query": "Parse this log and return IP and Error Code: '[2023-10-12 14:32:01] ERROR: Connection timeout from IP 192.168.1.55 (Code: 504)'",
        "solution": "IP:192.168.1.55, Code:504"
    },
    {
        "query": "Parse this log and return IP and Error Code: '[2023-10-13 09:11:22] FATAL: Memory leak detected from IP 10.0.0.5 (Code: 500)'",
        "solution": "IP:10.0.0.5, Code:500"
    },
    {
        "query": "Parse this log and return IP and Error Code: '[2023-10-14 08:30:00] ERROR: SSL handshake failed from IP 203.0.113.10 (Code: 495)'",
        "solution": "IP:203.0.113.10, Code:495"
    }
]

print("=" * 60)
print("GRADED MEMORY TEST: Extracting rule from 3 episodes...")
print("=" * 60)

rule = llm.extract_heuristic_rule(episodes)

if rule:
    conf = rule['confidence']
    grade = "REFLEX" if conf >= 0.95 else "HYBRID_SKILL" if conf >= 0.70 else "WEAK" if conf >= 0.40 else "DEAD"
    print(f"\nResult: {rule['pattern']}")
    print(f"Confidence: {conf:.2f}")
    print(f"Grade: {grade}")
    print(f"\nCode:\n{rule['python_code']}")
    
    # Quick sanity check
    print("\n--- Quick Execution Test ---")
    test = "Parse this log and return IP and Error Code: '[2024-01-01 00:00:00] ERROR: Unknown from IP 5.5.5.5 (Code: 418)'"
    result = safe_execute(rule['python_code'], test)
    print(f"Input:  {test[:80]}...")
    print(f"Output: {result}")
else:
    print("\nRule: None (robustness < 0.40, fully discarded)")
