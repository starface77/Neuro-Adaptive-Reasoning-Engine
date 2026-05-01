"""Fetch real GSM8K problems from HuggingFace, generate paraphrases, and convert to a_b_benchmark format."""

import json
import re
import urllib.request
import sys
import os

# Add project root to path so we can import nare.llm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nare import llm

def fetch_gsm8k(n: int = 30, offset: int = 0) -> list:
    """Download N GSM8K test problems from HuggingFace datasets API."""
    url = (
        f"https://datasets-server.huggingface.co/rows?"
        f"dataset=openai/gsm8k&config=main&split=test"
        f"&offset={offset}&length={n}"
    )
    print(f"Fetching {n} GSM8K problems from HuggingFace...")
    req = urllib.request.Request(url, headers={"User-Agent": "NARE-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    
    rows = data.get("rows", [])
    print(f"Got {len(rows)} rows")
    return rows


def extract_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K '#### NNNN' format."""
    m = re.search(r"####\s*(.+?)$", answer_text, re.MULTILINE)
    if m:
        return m.group(1).strip().replace(",", "")
    return ""


def generate_paraphrases(question: str, num_variants: int = 2) -> list:
    """Use the LLM to generate semantic paraphrases of the question."""
    prompt = (
        f"Here is a math word problem:\n\"{question}\"\n\n"
        f"Please rewrite this problem in {num_variants} different ways. "
        "Keep the exact same numbers, logic, and names, but change the phrasing, "
        "sentence structure, and vocabulary.\n\n"
        "Format EXACTLY like this, with nothing else:\n"
        "<p1>First paraphrase</p1>\n"
        "<p2>Second paraphrase</p2>"
    )
    try:
        # We use mode="REFLEX" to avoid <reasoning> blocks
        samples, _ = llm.generate_samples(prompt, n=1, temperature=0.7, mode="REFLEX")
        if not samples: return []
        text = samples[0].get("solution", "")
        
        variants = []
        for i in range(1, num_variants + 1):
            m = re.search(rf"<p{i}>(.*?)</p{i}>", text, re.DOTALL | re.IGNORECASE)
            if m:
                variants.append(m.group(1).strip())
        return variants
    except Exception as e:
        print(f"  Warning: failed to generate paraphrases: {e}")
        return []


def to_benchmark_format(rows: list, num_paraphrases: int = 2) -> list:
    """Convert HuggingFace GSM8K rows to a_b_benchmark task format."""
    tasks = []
    for i, row in enumerate(rows):
        r = row.get("row", row)
        question = r.get("question", "")
        answer_text = r.get("answer", "")
        
        numeric_answer = extract_answer(answer_text)
        if not numeric_answer:
            print(f"  Skipping row {i}: no numeric answer found")
            continue
        
        try:
            expected_nums = [float(numeric_answer)]
        except ValueError:
            print(f"  Skipping row {i}: can't parse '{numeric_answer}' as number")
            continue
            
        print(f"Processing [{i+1}/{len(rows)}]... generating paraphrases")
        paraphrases = generate_paraphrases(question, num_paraphrases)
        
        task = {
            "id": f"gsm8k_{i+1:03d}",
            "category": "GSM8K",
            "query": question,
            "paraphrases": paraphrases,
            "expected_solution": numeric_answer,
            "oracle_spec": {
                "type": "numeric_set",
                "expected": expected_nums,
            },
        }
        tasks.append(task)
    
    return tasks


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    rows = fetch_gsm8k(n, offset)
    tasks = to_benchmark_format(rows, num_paraphrases=2)
    
    out_path = "benchmarks/gsm8k_real.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main()
