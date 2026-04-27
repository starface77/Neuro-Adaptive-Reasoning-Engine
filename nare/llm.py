import os
import json
import re
import time
import urllib.request
import logging
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def _ensure_api_key():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

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
    _ensure_api_key()
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
    _ensure_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    
    if mode == "SLOW":
        system_prompt = """You are an advanced reasoning engine. 
REQUIRED FORMAT:
<abstract_signature>
[1-2 sentences categorizing the structural/mathematical/logical class of this problem. Example: "Mathematical sequence continuation. Polynomial finite differences." or "Text extraction. Log parsing for IPs."]
</abstract_signature>
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
            time.sleep(15)  # Rate limit spacing for heavy models like Gemma 27B
            
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
            
            a_match = re.search(r'<abstract_signature>(.*?)</abstract_signature>', content, re.DOTALL)
            abstract_signature = a_match.group(1).strip() if a_match else None
            
            # Final cleanup: if solution still contains XML tags (due to LLM structure error), strip them
            if "<" in solution and ">" in solution:
                solution = re.sub(r'<[^>]+>', '', solution).strip()
            
            samples.append({"solution": solution, "reasoning": reasoning, "abstract_signature": abstract_signature})
    
    return samples, total_tokens

def tree_of_thoughts(prompt: str, breadth: int = 3, depth: int = 2) -> tuple:
    """Tree-of-Thoughts: BFS with evaluation and backtracking.
    
    Generates a tree of reasoning paths by:
    1. Generating `breadth` initial thought branches
    2. Evaluating each branch for promise (0-10 score)
    3. Expanding only the top branches to next depth level
    4. Backtracking (pruning) branches that score below threshold
    
    Returns (best_candidates, total_tokens).
    """
    _ensure_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    total_tokens = 0
    
    # Phase 1: Generate initial thought branches
    branch_prompt = f"""You are performing Tree-of-Thoughts reasoning.

TASK: {prompt}

Generate {breadth} DIFFERENT initial reasoning approaches for this task.
For each approach, provide a brief thought (2-3 sentences) about how you would start solving it.

Format EXACTLY as:
<thought_1>[Your first approach]</thought_1>
<thought_2>[Your second approach]</thought_2>
<thought_3>[Your third approach]</thought_3>"""

    payload = {
        "contents": [{"parts": [{"text": branch_prompt}]}],
        "generationConfig": {"temperature": 0.9}
    }
    
    res = _post(url, payload)
    total_tokens += res.get('usageMetadata', {}).get('totalTokenCount', 0)
    
    content = ""
    if 'candidates' in res and res['candidates']:
        content = res['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '')
    
    # Parse thought branches
    thoughts = []
    for i in range(1, breadth + 1):
        match = re.search(rf'<thought_{i}>(.*?)</thought_{i}>', content, re.DOTALL)
        if match:
            thoughts.append(match.group(1).strip())
    
    if not thoughts:
        # Fallback: treat entire content as single thought
        thoughts = [content.strip()] if content.strip() else ["Direct approach"]
    
    # Phase 2: Evaluate each branch (score 0-10)
    time.sleep(15)
    eval_prompt = f"""TASK: {prompt}

Rate each reasoning approach on a scale of 0-10 for how promising it is:

"""
    for i, t in enumerate(thoughts):
        eval_prompt += f"Approach {i+1}: {t[:200]}\n\n"
    
    eval_prompt += """Respond with ONLY scores in this format:
<scores>[score1],[score2],[score3]</scores>"""
    
    payload = {
        "contents": [{"parts": [{"text": eval_prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    
    res = _post(url, payload)
    total_tokens += res.get('usageMetadata', {}).get('totalTokenCount', 0)
    
    eval_content = ""
    if 'candidates' in res and res['candidates']:
        eval_content = res['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '')
    
    # Parse scores
    scores = []
    score_match = re.search(r'<scores>(.*?)</scores>', eval_content, re.DOTALL)
    if score_match:
        try:
            scores = [float(s.strip()) for s in score_match.group(1).split(',')]
        except (ValueError, TypeError):
            scores = []
    
    if not scores:
        # Fallback: extract any numbers
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', eval_content)
        scores = [float(n) for n in nums[:len(thoughts)]]
    
    # Pad scores if needed
    while len(scores) < len(thoughts):
        scores.append(5.0)
    
    # Phase 3: Prune and expand top branches
    scored_thoughts = sorted(zip(scores, thoughts), reverse=True)
    # Keep top ceil(breadth/2) branches (backtrack/prune the rest)
    keep = max(1, (breadth + 1) // 2)
    top_branches = scored_thoughts[:keep]
    
    logging.info(f"[ToT] Scores: {[f'{s:.1f}' for s, _ in scored_thoughts]}")
    logging.info(f"[ToT] Keeping top {keep} branches, pruning {len(scored_thoughts)-keep}")
    
    # Phase 4: Deep expansion — develop winning branches into full solutions
    candidates = []
    for depth_step in range(depth):
        for score, thought in top_branches:
            time.sleep(15)
            expand_prompt = f"""You are an advanced reasoning engine performing Tree-of-Thoughts search.

TASK: {prompt}

SELECTED REASONING APPROACH (scored {score:.1f}/10):
{thought}

Now develop this approach into a COMPLETE solution.

REQUIRED FORMAT:
<abstract_signature>
[1-2 sentences categorizing the problem class]
</abstract_signature>
<reasoning>
[Full step-by-step solution following the selected approach]
</reasoning>
<solution>
[Your final answer]
</solution>"""

            payload = {
                "contents": [{"parts": [{"text": expand_prompt}]}],
                "generationConfig": {"temperature": 0.4}
            }
            
            res = _post(url, payload)
            total_tokens += res.get('usageMetadata', {}).get('totalTokenCount', 0)
            
            exp_content = ""
            if 'candidates' in res and res['candidates']:
                exp_content = res['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '')
            
            r_match = re.search(r'<reasoning>(.*?)</reasoning>', exp_content, re.DOTALL)
            s_match = re.search(r'<solution>(.*?)</solution>', exp_content, re.DOTALL)
            a_match = re.search(r'<abstract_signature>(.*?)</abstract_signature>', exp_content, re.DOTALL)
            
            reasoning = r_match.group(1).strip() if r_match else thought
            solution = s_match.group(1).strip() if s_match else exp_content.strip()
            abstract_sig = a_match.group(1).strip() if a_match else None
            
            if "<" in solution and ">" in solution:
                solution = re.sub(r'<[^>]+>', '', solution).strip()
            
            candidates.append({
                "solution": solution,
                "reasoning": reasoning,
                "abstract_signature": abstract_sig,
                "tot_score": score,
                "tot_depth": depth_step + 1,
            })
    
    logging.info(f"[ToT] Generated {len(candidates)} candidates, used {total_tokens} tokens")
    return candidates, total_tokens


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

def generate_stress_tests(episodes: list) -> list:
    """Generate ADVERSARIAL synthetic queries WITH LABELS to stress-test skills."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    
    prompt = "Analyze the following solved tasks. You must generate 10 NEW ADVERSARIAL tasks of the exact same category to stress-test our code.\n"
    prompt += "The 10 tasks MUST include:\n"
    prompt += "- 3 tasks with heavy text NOISE.\n"
    prompt += "- 3 tasks with BROKEN OR UNEXPECTED FORMATTING.\n"
    prompt += "- 2 tasks with MISSING FIELDS (should return Error gracefully).\n"
    prompt += "- 2 tasks with REORDERED FIELDS.\n\n"
    
    for i, ep in enumerate(episodes):
        prompt += f"Original Task: {ep['query']} -> Solution: {ep['solution']}\n"
    
    prompt += "\nOutput exactly 10 tasks in this EXACT format:\n"
    prompt += "TYPE: [POSITIVE or NEGATIVE]\n"
    prompt += "Q: [the query]\n"
    prompt += "S: [the expected correct solution, or 'IGNORE' if NEGATIVE]\n"
    prompt += "|||\n\n"
    prompt += "POSITIVE: Similar to originals. NEGATIVE: Tasks that look similar but belong to a DIFFERENT category (e.g. if originals are arithmetic sequences, a negative is a geometric one)."
    prompt += "Do not output anything else."
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }
    try:
        res = _post(url, payload)
        content = res['candidates'][0]['content']['parts'][0]['text']
        blocks = content.split("|||")
        
        tests = []
        for block in blocks:
            type_match = re.search(r'TYPE:\s*(POSITIVE|NEGATIVE)', block)
            q_match = re.search(r'Q:\s*(.*?)(?=\nS:|$)', block, re.DOTALL)
            s_match = re.search(r'S:\s*(.*)', block)
            if q_match and s_match:
                tests.append({
                    "type": type_match.group(1) if type_match else "POSITIVE",
                    "query": q_match.group(1).strip(),
                    "solution": s_match.group(1).strip()
                })
                
        return tests[:10]
    except Exception as e:
        logging.warning(f"Failed to generate adversarial stress tests: {e}")
        return []

def extract_heuristic_rule(
    episodes: list,
    oracle: "Oracle | None" = None,
    config: "NareConfig | None" = None,
):
    """Sleep Phase: Compress episodes into Executable Reflexes.

    Generates robust Python code with regex-based triggers and
    validates it against the original episodes + stress tests with
    refinement.

    ``oracle``: optional :class:`~nare.oracle.Oracle` used by
    :func:`_validate_skill` to judge correctness against verified
    solutions instead of the legacy string/numeric-overlap heuristic.
    Per-episode ``oracle_spec`` overrides this argument. When neither
    is supplied, the heuristic overlap fallback is used (documented as
    a fallback in :mod:`nare.oracle`).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    
    base_prompt = r"""You are a compiler that converts solved examples into a reusable Python STRUCTURAL SKILL.
Your goal is to extract the ABSTRACT LOGICAL STRUCTURE of the problem, not just regex match the exact text.

STRICT RULES FOR trigger(query: str) -> bool:
- This function must determine if a new query belongs to the EXACT SAME abstract structural class (e.g., 'is this a polynomial sequence continuation problem?' or 'is this a log parsing problem?').
- DO NOT just check for exact words from the training examples. Check for the structural properties (e.g., does it contain a sequence of numbers? does it ask for a 'total spent'?).
- Be robust: A math sequence might use commas or spaces. A financial log might use different names.

STRICT RULES FOR parse(query: str) -> dict:
- Extract all necessary variables from the raw text into a dictionary.
- Example: For sequences, extract the list of integers. For logs, extract IP and error code.

STRICT RULES FOR solve(vars: dict) -> str:
- Apply the generalized logic/math/algorithm to the extracted variables.
- MULTI-CASE REASONING: Your solver must be robust. For sequences, check if it's arithmetic, geometric, or quadratic by checking differences/ratios. For parsing, handle multiple optional fields.
- NEVER hardcode the answer. Implement the general formula.
- Return the final string exactly as the solution expects.

STRICT RULES FOR execute(query: str) -> str:
- Call parse() then solve(). Wrap in try/except. Return 'Error: <reason>' on failure.

Output EXACTLY this format and nothing else:

PATTERN: [Short structural name]

```python
import re
import math

def trigger(query: str) -> bool:
    # Heuristic checks for the abstract structural category
    # e.g., return bool(re.search(r'\d+,\s*\d+,\s*\d+', query)) and 'next term' in query.lower()
    return False

def parse(query: str) -> dict:
    # Extract variables robustly
    return {}

def solve(vars: dict) -> str:
    # Pure algorithmic solver
    return ""

def execute(query: str) -> str:
    try:
        vars = parse(query)
        result = solve(vars)
        return str(result)
    except Exception as e:
        return f'Error: {e}'
```

Here are the solved examples to learn from:

"""
    for i, ep in enumerate(episodes):
        base_prompt += f"Task {i+1}: {ep['query']}\nSolution {i+1}: {ep['solution']}\n\n"

    stress_tests = generate_stress_tests(episodes)
    all_test_cases = episodes + stress_tests

    attempts = 0
    current_prompt = base_prompt
    last_error = ""
    global_best_candidate = None

    while attempts < 3:
        attempts += 1
        
        candidates_generated = []
        logging.info(f"[Sleep] Attempt {attempts}: Generating population of 2 skill variants...")
        
        # Generate population with different temperatures for diversity
        import concurrent.futures
        
        def fetch_candidate(temp):
            payload = {
                "contents": [{"parts": [{"text": current_prompt}]}],
                "generationConfig": {"temperature": temp}
            }
            try:
                res = _post(url, payload)
                return res['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                logging.warning(f"Failed to generate candidate: {e}")
                return None
                
        # Generate sequentially to respect strict free tier rate limits (15 RPM)
        for temp in [0.2, 0.8]:
            content = fetch_candidate(temp)
            if content:
                candidates_generated.append(content)
            time.sleep(15) # Prevent 429 Rate Limits (1 request / 15s)
            
        if not candidates_generated:
            last_error = "API failure on all candidates."
            continue
            
        best_candidate = None
        best_overall = -1.0
        
        # Evaluate population
        for content in candidates_generated:
            pattern_match = re.search(r'PATTERN:\s*(.+)', content)
            pattern_name = pattern_match.group(1).strip() if pattern_match else "Unknown Pattern"
            
            code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
            python_code = code_match.group(1) if code_match else None
            
            if not python_code:
                continue
                
            scores, error_msg = _validate_skill(
                python_code, all_test_cases, oracle=oracle, config=config
            )
            overall = scores['overall']
            
            cand_data = {
                "pattern": pattern_name,
                "python_code": python_code,
                "confidence": overall,
                "trigger_accuracy": scores['trigger_accuracy'],
                "execute_accuracy": scores['execute_accuracy'],
                "error_msg": error_msg
            }
            
            if overall > best_overall:
                best_overall = overall
                best_candidate = cand_data
                
        if not best_candidate:
            logging.warning(f"[Sleep] Attempt {attempts}: No valid Python code produced by population.")
            last_error = "No valid Python code block found."
            current_prompt = base_prompt + f"\n\nYOUR PREVIOUS OUTPUT WAS INVALID.\nERROR: {last_error}\nPlease output exactly the required format."
            continue
            
        # Track global best across ALL attempts (never degrade)
        if not global_best_candidate or best_candidate['confidence'] > global_best_candidate['confidence']:
            global_best_candidate = best_candidate
            
        overall = global_best_candidate['confidence']
        pattern_name = global_best_candidate['pattern']
        python_code = global_best_candidate['python_code']
        error_msg = global_best_candidate['error_msg']
        
        if overall >= 0.95 or attempts == 3:
            logging.info(f"[Sleep] Promoting skill '{pattern_name}' (peak validation: {overall:.2f})")
            result = {k: v for k, v in global_best_candidate.items() if k != 'error_msg'}
            # CAP INITIAL CONFIDENCE: Strictly 0.70 max for new skills to enforce Shadow Mode
            result['confidence'] = min(0.70, overall)
            result['maturity'] = 0
            result['success_streak'] = 0
            return result
            
        # Refinement loop — use global best's diagnostics
        logging.warning(f"[Sleep] Best candidate across {attempts} attempt(s): overall={overall:.2f}")
        
        fix_instructions = []
        if global_best_candidate['trigger_accuracy'] < 0.90:
            fix_instructions.append(f"FIX TRIGGER (accuracy={global_best_candidate['trigger_accuracy']:.2f}): Your trigger() is missing valid inputs. Broaden the regex to catch ALL variations of this task type.")
        if global_best_candidate['execute_accuracy'] < 0.90:
            fix_instructions.append(f"FIX EXECUTE (accuracy={global_best_candidate['execute_accuracy']:.2f}): Your execute() fails on some inputs. Make your parsing more flexible with fallback regex patterns.")
        if "CRASH" in error_msg:
            fix_instructions.append("FIX CRASH: Add defensive checks. Never index arrays without checking len(). Always check regex match is not None.")
        
        fix_block = "\n".join(fix_instructions) if fix_instructions else "General improvement needed."
        
        current_prompt = base_prompt + f"\n\nYOUR BEST PREVIOUS CODE:\n```python\n{python_code}\n```\n\nDIAGNOSTIC REPORT:\n{error_msg}\n\nACTION REQUIRED:\n{fix_block}\n\nRewrite the ENTIRE skill. Output the corrected code in the exact format specified above."

    # === REJECTION: Only promote skills >= 0.40 ===
    if global_best_candidate and global_best_candidate['confidence'] >= 0.40:
        logging.info(f"[Sleep] Promoting skill '{global_best_candidate['pattern']}' (peak confidence: {global_best_candidate['confidence']:.2f})")
        logging.info(f"[Sleep] Diagnostics: trigger={global_best_candidate.get('trigger_accuracy', '?')}, exec={global_best_candidate.get('execute_accuracy', '?')}")
        result = {k: v for k, v in global_best_candidate.items() if k != 'error_msg'}
        return result

    peak = global_best_candidate['confidence'] if global_best_candidate else 0.0
    error_detail = global_best_candidate.get('error_msg', 'No details') if global_best_candidate else 'No candidate'
    logging.warning(f"[Sleep] REJECTED skill. Peak confidence {peak:.2f} < 0.40 — not stable enough.")
    logging.warning(f"[Sleep] Rejection reason: {error_detail[:500]}")
    return None


from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .oracle import Oracle  # noqa: F401
    from .config import NareConfig  # noqa: F401


def repair_skill(python_code: str, pattern: str, failing_tests: list,
                 error_msg: str, scores: dict, max_attempts: int = 2) -> str:
    """REM Sleep: Iteratively repair a skill that failed dream stress-tests.
    
    Uses LLM to analyze the failure diagnostics and generate a corrected
    version of the skill code.  Returns the repaired code, or the original
    code if repair fails.
    """
    _ensure_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"
    
    current_code = python_code
    
    for attempt in range(1, max_attempts + 1):
        # Build the failing test cases summary
        failing_summary = ""
        for t in failing_tests[:5]:
            failing_summary += f"  Q: {t.get('query', '')[:150]}\n"
            failing_summary += f"  Expected: {t.get('solution', '')[:100]}\n"
            failing_summary += f"  Type: {t.get('type', 'POSITIVE')}\n\n"
        
        prompt = f"""You are a code repair specialist. A compiled skill named '{pattern}' failed stress-testing during REM sleep.

CURRENT CODE:
```python
{current_code}
```

FAILURE DIAGNOSTICS:
- Trigger accuracy: {scores.get('trigger_accuracy', 0):.2f}
- Execute accuracy: {scores.get('execute_accuracy', 0):.2f}
- Overall score: {scores.get('overall', 0):.2f}
- Error details: {(error_msg or 'No specific errors')[:500]}

FAILING TEST CASES:
{failing_summary}

REPAIR INSTRUCTIONS:
1. Analyze WHY the code fails on the test cases above.
2. Fix the root cause (do not just patch individual cases).
3. Keep the same function signatures: trigger(query), parse(query), solve(vars), execute(query).
4. Make trigger() more robust to handle edge cases and noise.
5. Make solve() handle boundary conditions and unexpected inputs.

Output ONLY the corrected Python code inside ```python ... ``` tags. Nothing else."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3}
        }
        
        try:
            res = _post(url, payload)
            content = res.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            
            code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
            if code_match:
                repaired = code_match.group(1)
                logging.info(f"[REM Repair] Attempt {attempt}: generated repaired code ({len(repaired)} chars)")
                return repaired
        except Exception as e:
            logging.warning(f"[REM Repair] Attempt {attempt} failed: {e}")
    
    return python_code


def _validate_skill(
    python_code: str,
    episodes: list,
    oracle: "Oracle | None" = None,
    config: "NareConfig | None" = None,
) -> Tuple[dict, str]:
    """Validate a generated skill with decomposed, oracle-aware scoring.

    What ``overall`` is built from (weights live in
    :class:`nare.config.SkillValidationConfig`):

      * ``trigger_accuracy`` \u2014 fraction of *labelled originals* that
        ``trigger()`` returns True for. Real signal: the labels here
        come from solved episodes, not from the LLM.
      * ``execute_accuracy`` \u2014 fraction of *triggered originals* whose
        output is judged correct by an :class:`~nare.oracle.Oracle`.
        Each episode's own ``oracle_spec`` (if present) takes priority,
        then the explicit ``oracle`` argument, then a heuristic
        string/numeric-overlap fallback. The fallback reproduces the
        legacy behaviour but is documented as heuristic in
        :mod:`nare.oracle`.
      * ``negative_trap_accuracy`` \u2014 fraction of NEGATIVE stress
        tests on which ``trigger()`` correctly does NOT fire. Real
        signal regardless of label quality (the only thing checked is
        a boolean).
      * ``positive_no_crash_rate`` \u2014 advisory only by default.
        POSITIVE stress tests have LLM-generated labels and should not
        bias ``overall`` unless an external oracle vetoes them. With
        ``include_positive_stress=True`` and an ``oracle``, this
        signal is gated through the oracle and contributes
        ``w_positive_stress`` to overall.

    Returns ``(scores_dict, diagnostic_report)`` where ``scores_dict``
    contains keys: trigger_accuracy, execute_accuracy,
    negative_trap_accuracy, positive_no_crash_rate, overall.
    """
    from nare.sandbox import SecurityError, safe_load_module
    from nare.oracle import (
        build_oracle_from_spec,
        heuristic_overlap_oracle,
    )
    from nare.config import DEFAULT_CONFIG as _DEFAULT_CONFIG

    if config is None:
        config = _DEFAULT_CONFIG
    vcfg = config.skill_validation

    zero_scores = {
        "trigger_accuracy": 0.0,
        "execute_accuracy": 0.0,
        "negative_trap_accuracy": 0.0,
        "positive_no_crash_rate": 0.0,
        "overall": 0.0,
    }

    # AST validation + sandboxed loading goes through the single
    # canonical entry point in nare.sandbox. Any change to security
    # policy now lives in one place.
    try:
        safe_globals = safe_load_module(python_code)
    except SecurityError as e:
        return zero_scores, f"Code failed AST/Security check: {e}"
    except Exception as e:  # noqa: BLE001
        return zero_scores, f"Runtime error during compilation: {e}"

    if 'trigger' not in safe_globals or 'execute' not in safe_globals:
        return zero_scores, "Missing trigger() or execute() function."

    trigger_fn = safe_globals['trigger']
    execute_fn = safe_globals['execute']

    # Separate ORIGINAL episodes (have verified solutions) from STRESS
    # tests (LLM-generated; positive labels are not trustworthy).
    original_eps = [
        ep for ep in episodes if 'embedding' in ep or 'reasoning_trace' in ep
    ]
    stress_eps = [
        ep for ep in episodes
        if 'embedding' not in ep and 'reasoning_trace' not in ep
    ]

    def _oracle_for(ep: dict):
        spec = ep.get('oracle_spec')
        if spec:
            try:
                return build_oracle_from_spec(spec), 'episode oracle_spec'
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    f"[Validate] Bad oracle_spec on episode "
                    f"'{ep.get('query', '')[:40]}': {e}; falling back."
                )
        if oracle is not None:
            return oracle, 'caller-provided oracle'
        return (
            heuristic_overlap_oracle(ep.get('solution', '')),
            'heuristic overlap (fallback)',
        )

    # --- TRIGGER ACCURACY (on original episodes only) ---
    trigger_correct = 0
    trigger_total = len(original_eps) if original_eps else 1
    trigger_errors = []

    for ep in original_eps:
        try:
            if trigger_fn(ep['query']):
                trigger_correct += 1
            else:
                trigger_errors.append(
                    f"MISS: trigger() returned False on valid input: "
                    f"'{ep['query'][:60]}...'"
                )
        except Exception as e:
            trigger_errors.append(f"CRASH: trigger() crashed: {e}")

    trigger_accuracy = trigger_correct / trigger_total if trigger_total > 0 else 0.0

    # --- EXECUTE ACCURACY (oracle-checked against verified solutions) ---
    execute_correct = 0
    execute_total = 0
    execute_errors = []

    for ep in original_eps:
        try:
            if not trigger_fn(ep['query']):
                continue
            execute_total += 1

            result = execute_fn(ep['query'])
            result_str = str(result).strip()

            ep_oracle, source = _oracle_for(ep)
            ok, info = ep_oracle(ep['query'], result_str)
            if ok:
                execute_correct += 1
            else:
                execute_errors.append(
                    f"MISMATCH ({source}): "
                    f"q='{ep['query'][:40]}' -> '{result_str[:40]}': {info}"
                )
        except Exception as e:
            execute_total += 1
            execute_errors.append(
                f"CRASH: '{ep['query'][:40]}...' -> {e}"
            )

    execute_accuracy = (
        execute_correct / execute_total if execute_total > 0 else 0.0
    )

    # --- NEGATIVE TRAPS (real signal) + POSITIVE STRESS (advisory) ---
    negative_total = 0
    negative_pass = 0
    positive_total = 0
    positive_no_crash = 0
    positive_oracle_pass = 0
    positive_oracle_total = 0

    for ep in stress_eps:
        is_negative = ep.get('type') == 'NEGATIVE'
        try:
            triggered = trigger_fn(ep['query'])

            if is_negative:
                negative_total += 1
                if not triggered:
                    negative_pass += 1
                else:
                    execute_errors.append(
                        f"FALSE_POSITIVE: trigger() fired on TRAP: "
                        f"'{ep['query'][:40]}'"
                    )
                continue

            # POSITIVE stress: trigger first
            if not triggered:
                continue
            positive_total += 1

            result = execute_fn(ep['query'])
            result_str = str(result).strip()
            if not result_str.startswith("Error"):
                positive_no_crash += 1

            # Only count toward overall when the user opted in AND we
            # have an oracle that can adjudicate. Otherwise the LLM
            # label on this stress test is self-referential.
            if vcfg.include_positive_stress and oracle is not None:
                positive_oracle_total += 1
                ok, _info = oracle(ep['query'], result_str)
                if ok:
                    positive_oracle_pass += 1
        except Exception as e:
            if is_negative:
                negative_total += 1
            else:
                positive_total += 1
            execute_errors.append(
                f"STRESS_CRASH: '{ep['query'][:40]}...' -> {e}"
            )

    negative_trap_accuracy = (
        negative_pass / negative_total if negative_total > 0 else 1.0
    )
    positive_no_crash_rate = (
        positive_no_crash / positive_total if positive_total > 0 else 1.0
    )

    if vcfg.include_positive_stress and positive_oracle_total > 0:
        positive_stress_signal = positive_oracle_pass / positive_oracle_total
    else:
        positive_stress_signal = 0.0

    # --- OVERALL (weighted) ---
    raw_overall = (
        vcfg.w_trigger * trigger_accuracy
        + vcfg.w_execute * execute_accuracy
        + vcfg.w_negative_trap * negative_trap_accuracy
        + vcfg.w_positive_stress * positive_stress_signal
    )
    weight_sum = (
        vcfg.w_trigger
        + vcfg.w_execute
        + vcfg.w_negative_trap
        + (vcfg.w_positive_stress if vcfg.include_positive_stress else 0.0)
    )
    overall = raw_overall / weight_sum if weight_sum > 0 else 0.0

    # Hard gates: a skill that cannot trigger or execute on its own
    # training set is not promotable regardless of negative-trap luck.
    if (
        trigger_accuracy < vcfg.minimum_trigger_accuracy
        or execute_accuracy < vcfg.minimum_execute_accuracy
    ):
        overall = min(overall, 0.50)

    scores = {
        "trigger_accuracy": round(trigger_accuracy, 3),
        "execute_accuracy": round(execute_accuracy, 3),
        "negative_trap_accuracy": round(negative_trap_accuracy, 3),
        "positive_no_crash_rate": round(positive_no_crash_rate, 3),
        "overall": round(overall, 3),
    }

    if overall >= 0.99:
        return scores, ""

    report_lines = [
        f"SCORES: trigger={scores['trigger_accuracy']:.2f} | "
        f"execute={scores['execute_accuracy']:.2f} | "
        f"neg_trap={scores['negative_trap_accuracy']:.2f} | "
        f"pos_no_crash={scores['positive_no_crash_rate']:.2f} (advisory) | "
        f"overall={scores['overall']:.2f}"
    ]
    if trigger_errors:
        report_lines.append(f"[TRIGGER_ISSUES] ({len(trigger_errors)}):")
        for e in trigger_errors[:2]:
            report_lines.append(f"  - {e}")
    if execute_errors:
        report_lines.append(f"[EXECUTE_ISSUES] ({len(execute_errors)}):")
        for e in execute_errors[:3]:
            report_lines.append(f"  - {e}")

    return scores, "\n".join(report_lines)

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
    prompt += "The execute() function must work for ANY input of the same category (e.g., text parsing, math, logic). Use string manipulation or regex as needed.\n"
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
