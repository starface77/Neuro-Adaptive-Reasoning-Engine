import os
import json
import re
import time
import urllib.request
import logging
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def _post(url: str, payload: dict, retries: int = 5) -> dict:
    data = json.dumps(payload).encode('utf-8')
    
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode())
        except (urllib.error.HTTPError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                wait = 10 * (attempt + 1)
                logging.warning(f"Rate limit (429). Waiting {wait}s... (Attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif isinstance(e, TimeoutError):
                logging.warning(f"Timeout. Retrying in 5s... (Attempt {attempt+1}/{retries})")
                time.sleep(5)
            else:
                if hasattr(e, 'read'):
                    logging.error(f"HTTPError {e.code}: {e.read().decode()}")
                raise e
    
    raise Exception("Max retries exceeded for API request.")

def get_embedding(text: str) -> list:
    """Compute embedding via Gemini gemini-embedding-001 (dim=3072)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={API_KEY}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    response = _post(url, payload)
    return response['embedding']['values']

def generate_samples(prompt: str, n: int = 3, temperature: float = 0.8, mode: str = "SLOW"):
    """
    Generate N candidates. 
    mode can be 'SLOW', 'HYBRID', or 'REFLEX'. The mode strictly controls the required XML output structure.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    
    if mode == "SLOW":
        system_prompt = """You are an advanced reasoning engine. 
REQUIRED FORMAT:
<reasoning>
[Your step-by-step logical trace, breaking down the problem from scratch]
</reasoning>
<solution>
[Your final actionable answer or code]
</solution>"""
    elif mode == "HYBRID":
        system_prompt = """You are an advanced reasoning engine performing DELTA REASONING.
You have been provided with a past solution. DO NOT reason from scratch.
REQUIRED FORMAT:
<delta_reasoning>
[MAX 2 SENTENCES. Describe ONLY the logical differences or delta from the past solution]
</delta_reasoning>
<solution>
[Your final actionable answer or code, adapted for the new task]
</solution>"""
    elif mode == "REFLEX":
        system_prompt = """You are an advanced execution engine. 
A specific semantic rule/skill has been triggered. DO NOT PERFORM DEDUCTIVE REASONING.
REQUIRED FORMAT:
<rule_activation>
[Name of the rule being applied]
</rule_activation>
<solution>
[Immediate actionable answer or code following the rule exactly]
</solution>"""

    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": temperature}
    }
    
    samples = []
    total_tokens = 0
    
    for i in range(n):
        if i > 0:
            time.sleep(4)  # Rate limit spacing
            
        res = _post(url, payload)
        if 'candidates' in res and res['candidates']:
            cand = res['candidates'][0]
            if 'content' not in cand or 'parts' not in cand['content']:
                continue
            content = cand['content']['parts'][0]['text']
            total_tokens += res.get('usageMetadata', {}).get('totalTokenCount', 0)
            
            reasoning, solution = "No trace provided.", content
            r_match = None
            
            if mode == "SLOW":
                r_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
            elif mode == "HYBRID":
                r_match = re.search(r'<delta_reasoning>(.*?)</delta_reasoning>', content, re.DOTALL)
            elif mode == "REFLEX":
                r_match = re.search(r'<rule_activation>(.*?)</rule_activation>', content, re.DOTALL)
                
            if r_match: reasoning = r_match.group(1).strip()
                
            s_match = re.search(r'<solution>(.*?)</solution>', content, re.DOTALL)
            if s_match: solution = s_match.group(1).strip()
            elif r_match: solution = content.replace(r_match.group(0), "").strip()
            
            samples.append({"solution": solution, "reasoning": reasoning})
    
    return samples, total_tokens

def llm_pairwise_judge(query: str, sol_a: str, sol_b: str) -> int:
    """Returns 1 if A is better, 2 if B is better."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    prompt = f"""You are an objective judge evaluating two solutions to a task.
Task: {query}

Candidate A: {sol_a}
Candidate B: {sol_b}

Evaluate correctness, completeness, and lack of hallucinations.
Output strictly 'A' or 'B' on the final line."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0}
    }
    
    res = _post(url, payload)
    content = res['candidates'][0]['content']['parts'][0]['text'].strip().upper()
    last_word = content.split()[-1]
    last_word = re.sub(r'[^AB]', '', last_word)
    
    return 1 if last_word == 'A' else 2

def extract_heuristic_rule(episodes: list):
    """Sleep Phase: Compress episodes into Executable Reflexes."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    prompt = "Analyze these past solved tasks and extract an EXECUTABLE SKILL.\n"
    prompt += "You must output the rule exactly in this markdown format:\n\n"
    prompt += "PATTERN: [High-level name of the pattern]\n\n"
    prompt += "```python\n"
    prompt += "def trigger(query: str) -> bool:\n"
    prompt += "    # Return True if this skill applies to the query, else False\n"
    prompt += "    return ...\n\n"
    prompt += "def execute(query: str) -> str:\n"
    prompt += "    # Execute the algorithm and return the EXACT final answer as a string.\n"
    prompt += "    return ...\n"
    prompt += "```\n\n"
    
    for i, ep in enumerate(episodes):
        prompt += f"Task {i+1}: {ep['query']}\nSolution {i+1}: {ep['solution']}\n\n"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    
    res = _post(url, payload)
    content = res['candidates'][0]['content']['parts'][0]['text']
    
    pattern_match = re.search(r'PATTERN:\s*(.+)', content)
    pattern_name = pattern_match.group(1).strip() if pattern_match else "Unknown Pattern"
    
    code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    python_code = code_match.group(1) if code_match else "def trigger(q): return False\ndef execute(q): return 'Error'"
            
    return {
        "pattern": pattern_name,
        "python_code": python_code,
        "confidence": 0.75
    }

def merge_heuristic_rules(existing_rule: dict, new_episodes: list):
    """Refine an existing Executable Reflex with new episodes."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    
    prompt = "You are consolidating knowledge. We have an EXISTING RULE and NEW EPISODES that match it.\n"
    prompt += f"EXISTING PATTERN: {existing_rule.get('pattern')}\n"
    prompt += f"EXISTING CODE:\n```python\n{existing_rule.get('python_code')}\n```\n\n"
    prompt += "NEW EPISODES:\n"
    for i, ep in enumerate(new_episodes):
        prompt += f"Ep {i+1}: {ep['query']} -> {ep['solution'][:200]}...\n"
        
    prompt += "\nRefine the EXISTING RULE to be more general and robust. Update the trigger logic or execute logic if necessary.\n"
    prompt += "You must output the rule exactly in this markdown format:\n\n"
    prompt += "PATTERN: [Refined pattern name]\n\n"
    prompt += "```python\n"
    prompt += "def trigger(query: str) -> bool:\n"
    prompt += "    return ...\n\n"
    prompt += "def execute(query: str) -> str:\n"
    prompt += "    return ...\n"
    prompt += "```\n"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    
    res = _post(url, payload)
    content = res['candidates'][0]['content']['parts'][0]['text']
    
    pattern_match = re.search(r'PATTERN:\s*(.+)', content)
    pattern_name = pattern_match.group(1).strip() if pattern_match else existing_rule.get('pattern', 'Merged Pattern')
    
    code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    if code_match:
        existing_rule['pattern'] = pattern_name
        existing_rule['python_code'] = code_match.group(1)
        
    return existing_rule
