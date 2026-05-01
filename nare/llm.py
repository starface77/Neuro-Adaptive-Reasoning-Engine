import os
import json
import re
import time
import urllib.request
import logging
import random
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .oracle import Oracle
    from .config import NareConfig

# Anthropic API configuration via environment variables
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:20128/v1")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_API_KEY", "sk-1703362f337860d0-684d06-26299dc8")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "kr/claude-sonnet-4.5")

def _ensure_api_key():
    """Ensure API key is configured."""
    if not ANTHROPIC_AUTH_TOKEN:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Set it in your .env file or export it in your shell. "
            "For local proxy, the default token will be used if not set."
        )

def _post_anthropic(endpoint: str, payload: dict, retries: int = 5) -> str:
    """POST to Anthropic API via local proxy and parse SSE stream."""
    url = f"{ANTHROPIC_BASE_URL}/{endpoint}"
    data = json.dumps(payload).encode('utf-8')

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_AUTH_TOKEN,
        'anthropic-version': '2023-06-01'
    }

    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                # Parse SSE stream
                content_parts = []
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            event_data = json.loads(line[6:])
                            if event_data.get('type') == 'content_block_delta':
                                delta = event_data.get('delta', {})
                                text = delta.get('text', '')
                                if text:
                                    content_parts.append(text)
                        except json.JSONDecodeError:
                            continue

                return ''.join(content_parts)

        except urllib.error.HTTPError as e:
            if e.code == 429:
                base_wait = min(10 * (2 ** attempt), 60)
                jitter = random.uniform(0, base_wait * 0.1)
                wait_time = base_wait + jitter
                logging.warning(f"Rate limit (429). Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("Max retries exceeded for API request.")
            else:
                raise
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait_time = min(5 * (2 ** attempt), 30)
            logging.warning(f"Anthropic API error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)

    raise Exception("Max retries exceeded for Anthropic API")

# Global embedding model (lazy loaded)
_embedding_model = None

def get_embedding(text: str) -> list:
    """Compute embedding via sentence-transformers (3072-dim)."""
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        # Use a model that produces 3072-dim embeddings
        # If not available, we'll use a smaller model and pad
        try:
            _embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            logging.info("[Embeddings] Loaded BAAI/bge-large-en-v1.5 (1024-dim)")
        except:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("[Embeddings] Loaded all-MiniLM-L6-v2 (384-dim)")

    # Get embedding
    emb = _embedding_model.encode(text, convert_to_numpy=True)

    # Pad to 3072 dimensions if needed
    if len(emb) < 3072:
        import numpy as np
        padding = np.zeros(3072 - len(emb))
        emb = np.concatenate([emb, padding])
    elif len(emb) > 3072:
        emb = emb[:3072]

    return emb.tolist()

def generate_samples(prompt: str, n: int = 3, temperature: float = 0.8, mode: str = "SLOW"):
    """Generate N candidates using Anthropic API."""
    _ensure_api_key()

    if mode == "SLOW":
        system_prompt = """You are an advanced reasoning engine.
REQUIRED FORMAT:
<abstract_signature>
[1-2 sentences categorizing the structural/mathematical/logical class of this problem]
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
    else:  # FAST or any other mode
        system_prompt = """You are a fast reasoning engine. Provide a direct answer.
REQUIRED FORMAT:
<solution>
[Your answer]
</solution>"""

    samples = []
    total_tokens = 0

    logging.info(f"[LLM] generate_samples called: mode={mode}, n={n}, temp={temperature}")

    for i in range(n):
        if i > 0:
            time.sleep(2)  # Rate limit spacing

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            content = _post_anthropic("messages", payload)
            total_tokens += len(content.split())  # Rough estimate

            reasoning, solution = "No trace provided.", content
            r_match = None

            if mode == "SLOW":
                r_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
            elif mode == "HYBRID":
                r_match = re.search(r'<delta_reasoning>(.*?)</delta_reasoning>', content, re.DOTALL)
            elif mode == "REFLEX":
                r_match = re.search(r'<rule_activation>(.*?)</rule_activation>', content, re.DOTALL)

            if r_match:
                reasoning = r_match.group(1).strip()

            s_match = re.search(r'<solution>(.*?)</solution>', content, re.DOTALL)
            if s_match:
                solution = s_match.group(1).strip()
            elif r_match:
                solution = content.replace(r_match.group(0), "").strip()

            a_match = re.search(r'<abstract_signature>(.*?)</abstract_signature>', content, re.DOTALL)
            abstract_signature = a_match.group(1).strip() if a_match else None

            if "<" in solution and ">" in solution:
                solution = re.sub(r'<[^>]+>', '', solution).strip()

            samples.append({"solution": solution, "reasoning": reasoning, "abstract_signature": abstract_signature})
        except Exception as e:
            logging.warning(f"[LLM] Failed to generate sample {i+1}: {e}")

    logging.info(f"[LLM] Returning {len(samples)} samples, ~{total_tokens} tokens")
    return samples, total_tokens

def tree_of_thoughts(prompt: str, breadth: int = 1, depth: int = 2) -> tuple:
    """Best-of-N reasoning (simplified for Anthropic)."""
    # For now, just generate breadth samples with varying temperature
    all_candidates = []
    total_tokens = 0

    temps = [0.5][:breadth]  # Single temperature for speed
    for temp in temps:
        candidates, tokens = generate_samples(prompt, n=1, temperature=temp, mode="SLOW")
        all_candidates.extend(candidates)
        total_tokens += tokens
        time.sleep(2)

    return all_candidates, total_tokens

def best_of_n_with_prescore(prompt: str, breadth: int = 1) -> tuple:
    """Best-of-N candidate generation."""
    return tree_of_thoughts(prompt, breadth=breadth, depth=1)

def llm_pairwise_judge(query: str, sol_a: str, sol_b: str) -> int:
    """Returns 1 if A is better, 2 if B is better."""
    prompt = f"""You are an objective judge evaluating two solutions to a task.
Task: {query}

Candidate A: {sol_a}
Candidate B: {sol_b}

Evaluate correctness, completeness, and lack of hallucinations.
Output strictly 'A' or 'B' on the final line."""

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        content = _post_anthropic("messages", payload)
        last_word = content.strip().split()[-1].upper()
        last_word = re.sub(r'[^AB]', '', last_word)
        return 1 if last_word == 'A' else 2
    except:
        return 1

def generate_stress_tests(episodes: list) -> list:
    """Generate ADVERSARIAL synthetic queries WITH LABELS to stress-test skills."""
    prompt = "Analyze the following solved tasks. You must generate 10 NEW ADVERSARIAL tasks of the exact same category to stress-test our code.\n"
    prompt += "The 10 tasks MUST include:\n"
    prompt += "- 3 tasks with heavy text NOISE.\n"
    prompt += "- 3 tasks with BROKEN OR UNEXPECTED FORMATTING.\n"
    prompt += "- 2 tasks with MISSING FIELDS (should return Error gracefully).\n"
    prompt += "- 2 tasks with REORDERED FIELDS.\n\n"

    for i, ep in enumerate(episodes[:3]):  # Limit to 3 examples
        prompt += f"Original Task: {ep['query'][:200]} -> Solution: {ep['solution'][:100]}\n"

    prompt += "\nOutput exactly 10 tasks in this EXACT format:\n"
    prompt += "TYPE: [POSITIVE or NEGATIVE]\n"
    prompt += "Q: [the query]\n"
    prompt += "S: [the expected correct solution, or 'IGNORE' if NEGATIVE]\n"
    prompt += "|||\n\n"

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        content = _post_anthropic("messages", payload)
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

def extract_heuristic_rule(episodes: list, oracle: Optional["Oracle"] = None, config: Optional["NareConfig"] = None):
    """Sleep Phase: Compress episodes into Executable Reflexes."""
    if len(episodes) < 2:
        return None

    # Build prompt for skill extraction
    prompt = """You are a compiler that converts solved examples into a reusable Python skill.

Extract the ABSTRACT LOGICAL STRUCTURE of the problem.

STRICT RULES:
- trigger(query: str) -> bool: Check if query belongs to the same structural class
- parse(query: str) -> dict: Extract variables from text
- solve(vars: dict) -> str: Apply the general algorithm
- execute(query: str) -> str: Call parse() then solve()

Output EXACTLY this format:

PATTERN: [Short structural name]

```python
import re

def trigger(query: str) -> bool:
    # Check structural properties
    return False

def parse(query: str) -> dict:
    # Extract variables
    return {}

def solve(vars: dict) -> str:
    # Apply algorithm
    return ""

def execute(query: str) -> str:
    try:
        vars = parse(query)
        result = solve(vars)
        return str(result)
    except Exception as e:
        return f'Error: {e}'
```

Examples to learn from:

"""
    for i, ep in enumerate(episodes[:3]):
        prompt += f"Task {i+1}: {ep['query'][:200]}\nSolution {i+1}: {ep['solution'][:100]}\n\n"

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        content = _post_anthropic("messages", payload)

        pattern_match = re.search(r'PATTERN:\s*(.+)', content)
        pattern_name = pattern_match.group(1).strip() if pattern_match else "Unknown Pattern"

        code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        python_code = code_match.group(1) if code_match else None

        if not python_code:
            logging.warning("[Sleep] No valid Python code in LLM response")
            return None

        # Quick validation
        scores, error_msg = _validate_skill(python_code, episodes, oracle=oracle, config=config)
        overall = scores['overall']

        if overall < 0.40:
            logging.warning(f"[Sleep] Skill rejected: overall={overall:.2f} < 0.40")
            return None

        logging.info(f"[Sleep] Promoting skill '{pattern_name}' (confidence: {overall:.2f})")

        return {
            "pattern": pattern_name,
            "python_code": python_code,
            "confidence": min(0.70, overall),  # Cap at 0.70 for shadow mode
            "maturity": 0,
            "success_streak": 0,
            "trigger_accuracy": scores['trigger_accuracy'],
            "execute_accuracy": scores['execute_accuracy']
        }

    except Exception as e:
        logging.warning(f"[Sleep] Failed to extract skill: {e}")
        return None

def repair_skill(python_code: str, pattern: str, failing_tests: list,
                 error_msg: str, scores: dict, max_attempts: int = 2,
                 validator=None, baseline_score: Optional[float] = None) -> str:
    """REM Sleep: Iteratively repair a skill."""

    best_code = python_code
    best_score = baseline_score if baseline_score is not None else scores.get("overall", 0.0)

    for attempt in range(1, max_attempts + 1):
        failing_summary = ""
        for t in failing_tests[:3]:
            failing_summary += f"  Q: {t.get('query', '')[:100]}\n"
            failing_summary += f"  Expected: {t.get('solution', '')[:50]}\n\n"

        prompt = f"""You are a code repair specialist. A skill named '{pattern}' failed testing.

CURRENT CODE:
```python
{python_code}
```

FAILURE DIAGNOSTICS:
- Trigger accuracy: {scores.get('trigger_accuracy', 0):.2f}
- Execute accuracy: {scores.get('execute_accuracy', 0):.2f}
- Overall: {scores.get('overall', 0):.2f}

FAILING CASES:
{failing_summary}

Fix the root cause. Keep function signatures unchanged.
Output ONLY the corrected Python code inside ```python ... ``` tags."""

        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 2048,
            "temperature": 0.3 + (attempt * 0.2),
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            content = _post_anthropic("messages", payload)
            code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)

            if code_match:
                repaired = code_match.group(1)
                logging.info(f"[REM Repair] Attempt {attempt}/{max_attempts}: generated repair")

                if validator is None:
                    return repaired

                try:
                    cand_score = float(validator(repaired))
                except Exception as e:
                    logging.warning(f"[REM Repair] Validator failed: {e}")
                    cand_score = -1.0

                logging.info(f"[REM Repair] Attempt {attempt} score: {cand_score:.3f} (baseline: {best_score:.3f})")

                if cand_score > best_score:
                    best_code = repaired
                    best_score = cand_score
                    python_code = repaired

                if cand_score >= 0.80:
                    logging.info(f"[REM Repair] Early-stop at score {cand_score:.3f}")
                    return best_code

        except Exception as e:
            logging.warning(f"[REM Repair] Attempt {attempt} failed: {e}")

    if validator is not None and best_code != python_code:
        return best_code
    return python_code

def _validate_skill(python_code: str, episodes: list, oracle: Optional["Oracle"] = None, config: Optional["NareConfig"] = None) -> Tuple[dict, str]:
    """Validate a generated skill."""
    from nare.sandbox import SecurityError, safe_load_module
    from nare.oracle import build_oracle_from_spec, cached_episode_oracle
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

    try:
        safe_globals = safe_load_module(python_code)
    except SecurityError as e:
        return zero_scores, f"Code failed AST/Security check: {e}"
    except Exception as e:
        return zero_scores, f"Runtime error during compilation: {e}"

    if 'trigger' not in safe_globals or 'execute' not in safe_globals:
        return zero_scores, "Missing trigger() or execute() function."

    trigger_fn = safe_globals['trigger']
    execute_fn = safe_globals['execute']

    original_eps = [ep for ep in episodes if 'embedding' in ep or 'reasoning_trace' in ep]
    stress_eps = [ep for ep in episodes if 'embedding' not in ep and 'reasoning_trace' not in ep]

    def _oracle_for(ep: dict):
        spec = ep.get('oracle_spec')
        if spec:
            try:
                return build_oracle_from_spec(spec), 'episode oracle_spec'
            except Exception as e:
                logging.warning(f"[Validate] Bad oracle_spec: {e}")
        if oracle is not None:
            return oracle, 'caller-provided oracle'
        return cached_episode_oracle(ep.get('solution', '')), 'cached episode'

    trigger_correct = 0
    trigger_total = len(original_eps) if original_eps else 1
    trigger_errors = []

    for ep in original_eps:
        try:
            if trigger_fn(ep['query']):
                trigger_correct += 1
            else:
                trigger_errors.append(f"MISS: '{ep['query'][:60]}...'")
        except Exception as e:
            trigger_errors.append(f"CRASH: {e}")

    trigger_accuracy = trigger_correct / trigger_total if trigger_total > 0 else 0.0

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
                execute_errors.append(f"MISMATCH ({source}): '{result_str[:40]}': {info}")
        except Exception as e:
            execute_total += 1
            execute_errors.append(f"CRASH: {e}")

    execute_accuracy = execute_correct / execute_total if execute_total > 0 else 0.0

    negative_total = 0
    negative_pass = 0
    positive_total = 0
    positive_no_crash = 0

    for ep in stress_eps:
        is_negative = ep.get('type') == 'NEGATIVE'
        try:
            triggered = trigger_fn(ep['query'])

            if is_negative:
                negative_total += 1
                if not triggered:
                    negative_pass += 1
                continue

            if not triggered:
                continue
            positive_total += 1

            result = execute_fn(ep['query'])
            result_str = str(result).strip()
            if not result_str.startswith("Error"):
                positive_no_crash += 1
        except Exception as e:
            if is_negative:
                negative_total += 1
            else:
                positive_total += 1

    negative_trap_accuracy = negative_pass / negative_total if negative_total > 0 else 1.0
    positive_no_crash_rate = positive_no_crash / positive_total if positive_total > 0 else 1.0

    overall = (
        vcfg.w_trigger * trigger_accuracy
        + vcfg.w_execute * execute_accuracy
        + vcfg.w_negative_trap * negative_trap_accuracy
    )
    weight_sum = vcfg.w_trigger + vcfg.w_execute + vcfg.w_negative_trap
    overall = overall / weight_sum if weight_sum > 0 else 0.0

    if trigger_accuracy < vcfg.minimum_trigger_accuracy or execute_accuracy < vcfg.minimum_execute_accuracy:
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
        f"overall={scores['overall']:.2f}"
    ]
    if trigger_errors:
        report_lines.append(f"[TRIGGER_ISSUES] ({len(trigger_errors)})")
    if execute_errors:
        report_lines.append(f"[EXECUTE_ISSUES] ({len(execute_errors)})")

    return scores, "\n".join(report_lines)

def merge_heuristic_rules(existing_rule: dict, new_episodes: list):
    """Refine an existing Executable Reflex with new episodes."""
    logging.info("[Sleep] merge_heuristic_rules not implemented for Anthropic yet")
    return existing_rule
