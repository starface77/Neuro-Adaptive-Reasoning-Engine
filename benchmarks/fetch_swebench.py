"""Fetch SWE-bench Lite problems from HuggingFace and convert to a_b_benchmark format."""

import json
import re
import urllib.request
import sys
import os

def fetch_swebench(n: int = 10, offset: int = 0) -> list:
    """Download N SWE-bench Lite problems from HuggingFace datasets API."""
    all_rows = []
    chunk_size = 100
    
    print(f"Fetching {n} SWE-bench problems from HuggingFace in chunks of {chunk_size}...")
    
    while n > 0:
        current_batch = min(n, chunk_size)
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=princeton-nlp/SWE-bench_Lite&config=default&split=test"
            f"&offset={offset}&length={current_batch}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NARE-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            
            rows = data.get("rows", [])
            all_rows.extend(rows)
            
            if len(rows) < current_batch:
                break # No more rows available
                
            offset += current_batch
            n -= current_batch
        except Exception as e:
            print(f"Error fetching chunk: {e}")
            break
            
    print(f"Got {len(all_rows)} total rows")
    return all_rows

def extract_modified_files(patch: str) -> list:
    """Extract list of modified files from a git patch."""
    files = []
    for line in patch.split('\n'):
        if line.startswith('--- a/'):
            files.append(line[6:].strip())
    return files

def to_benchmark_format(rows: list) -> list:
    """Convert HuggingFace SWE-bench rows to a_b_benchmark task format."""
    tasks = []
    for i, row in enumerate(rows):
        r = row.get("row", row)
        instance_id = r.get("instance_id", f"swe_{i}")
        problem = r.get("problem_statement", "")
        patch = r.get("patch", "")
        
        # For a local benchmark without full docker environments, we use a proxy oracle:
        # The agent must successfully identify the files that need to be modified
        # or output a patch that contains the critical changes.
        modified_files = extract_modified_files(patch)
        if not modified_files:
            continue
            
        # We ask the LLM to output a python script that prints the files to modify,
        # or just output the patch. For simplicity in our current sandbox, we'll
        # ask it to identify the files that need fixing.
        query = (
            f"Given the following GitHub issue, identify the exact file paths "
            f"that need to be modified to fix the issue. Print them one per line.\n\n"
            f"Issue: {problem[:2000]}..." # Truncate for the prompt
        )
        
        task = {
            "id": instance_id,
            "category": "SWE-bench",
            "query": query,
            "paraphrases": [
                f"Which files must be edited to resolve this bug report?\n\nReport: {problem[:2000]}..."
            ],
            "expected_solution": modified_files[0],
            "oracle_spec": {
                "type": "string_contains",
                "must_contain": [modified_files[0]]
            },
        }
        tasks.append(task)
    
    return tasks


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    rows = fetch_swebench(n, offset)
    tasks = to_benchmark_format(rows)
    
    out_path = "benchmarks/swebench_real.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(tasks)} tasks to {out_path}")


if __name__ == "__main__":
    main()
