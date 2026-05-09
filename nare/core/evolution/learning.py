"""Library Learning: Rule Discovery through Search

NOT extraction - SEARCH over rule space.

Pipeline:
1. Rule Sampler: Generate N candidate rules (LLM proposals)
2. Compiler: Convert rules to executable operators
3. Evaluator: Test on holdout + perturbations
4. Selection: Pick best generalizing rule
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from ...reasoning import llm
from ...execution.sandboxes.base import safe_load_module, SecurityError

def discover_rule(
    episodes: List[Dict[str, Any]],
    oracle: Optional[Callable] = None,
    n_candidates: int = 5,
    holdout_ratio: float = 0.3
) -> Optional[Dict[str, Any]]:
    """Discover generalizing rule through search.

    Args:
        episodes: Training episodes (query, solution pairs)
        oracle: Verification function
        n_candidates: Number of rule candidates to sample
        holdout_ratio: Fraction of episodes for holdout test

    Returns:
        Best rule dict or None if no rule generalizes
    """

    if len(episodes) < 3:
        logging.info("[Library Learning] Need ≥3 episodes for rule discovery")
        return None

    n_holdout = max(1, int(len(episodes) * holdout_ratio))
    episodes = list(episodes)  # defensive copy before shuffle
    np.random.shuffle(episodes)
    train_eps = episodes[:-n_holdout]
    holdout_eps = episodes[-n_holdout:]

    logging.info(f"[Library Learning] Train: {len(train_eps)}, Holdout: {len(holdout_eps)}")

    candidates = sample_candidate_rules(train_eps, n=n_candidates)
    if not candidates:
        logging.warning("[Library Learning] No candidates generated")
        return None

    logging.info(f"[Library Learning] Generated {len(candidates)} candidate rules")

    operators = []
    for i, cand in enumerate(candidates):
        op = compile_rule_to_operator(cand)
        if op:
            operators.append((i, cand, op))

    if not operators:
        logging.warning("[Library Learning] No operators compiled successfully")
        return None

    logging.info(f"[Library Learning] Compiled {len(operators)} operators")

    scores = []
    for idx, cand, op in operators:
        score = evaluate_operator(
            operator=op,
            holdout_eps=holdout_eps,
            train_eps=train_eps,
            oracle=oracle
        )
        scores.append((idx, score))
        logging.info(f"[Library Learning] Candidate {idx}: score={score:.2f}")

    if not scores:
        return None

    best_pos = max(range(len(scores)), key=lambda j: scores[j][1])
    best_idx, best_score = scores[best_pos]

    if best_score < 0.6:
        logging.warning(f"[Library Learning] Best score {best_score:.2f} < 0.6, reject")
        return None

    _, best_cand, best_op = operators[best_pos]

    logging.info(f"[Library Learning] Selected rule with score {best_score:.2f}")

    return {
        'pattern': best_cand['pattern'],
        'python_code': best_cand['code'],
        'trigger_fn': best_op['trigger'],
        'execute_fn': best_op['execute'],
        'confidence': best_score,
        'train_size': len(train_eps),
        'holdout_score': best_score
    }

def sample_candidate_rules(episodes: List[Dict], n: int = 5) -> List[Dict]:
    """Sample N candidate rules using LLM.

    LLM proposes multiple rule hypotheses, not "the" rule.
    """

    prompt = f"""
Each rule should:
1. Define a PATTERN (when does this rule apply?)
2. Define an ACTION (what transformation to perform?)
3. Be GENERAL (work on variations, not just these examples)

Examples:
"""

    for i, ep in enumerate(episodes[:3]):
        prompt += f"\nExample {i+1}:\n"
        prompt += f"  Query: {ep['query']}\n"
        prompt += f"  Solution: {ep.get('solution', ep.get('answer', ''))}\n"

    prompt += f"""
Generate {n} DIFFERENT candidate rules. For each rule, provide:

RULE 1:
PATTERN: [When does this apply? Be specific about input structure]
ACTION: [What transformation? Be algorithmic]
CODE:
```python
def trigger(query: str) -> bool:
    # Return True if this rule applies to query
    pass

def execute(query: str) -> str:
    # Perform the transformation
    pass
```

RULE 2:
...

Make rules DIVERSE - try different generalizations.
"""

    try:
        response = llm._post_anthropic("messages", {
            "model": llm.ANTHROPIC_MODEL,
            "max_tokens": 4096,
            "temperature": 0.8,
            "messages": [{"role": "user", "content": prompt}]
        })

        candidates = parse_candidate_rules(response)
        return candidates[:n]

    except Exception as e:
        logging.error(f"[Library Learning] Failed to sample rules: {e}")
        return []

def parse_candidate_rules(response: str) -> List[Dict]:
    """Parse multiple rule candidates from LLM response."""
    import re

    candidates = []

    rule_blocks = re.split(r'RULE \d+:', response)

    for block in rule_blocks[1:]:

        pattern_match = re.search(r'PATTERN:\s*(.+?)(?=ACTION:|CODE:|$)', block, re.DOTALL)
        pattern = pattern_match.group(1).strip() if pattern_match else "Unknown"

        code_match = re.search(r'```python\n(.*?)\n```', block, re.DOTALL)
        if not code_match:
            continue

        code = code_match.group(1)

        candidates.append({
            'pattern': pattern,
            'code': code
        })

    return candidates

def compile_rule_to_operator(candidate: Dict) -> Optional[Dict]:
    """Compile rule code into executable operator.

    Returns dict with trigger_fn and execute_fn, or None if compilation fails.
    """

    try:
        safe_globals = safe_load_module(candidate['code'])

        if 'trigger' not in safe_globals or 'execute' not in safe_globals:
            logging.warning("[Library Learning] Missing trigger() or execute()")
            return None

        return {
            'trigger': safe_globals['trigger'],
            'execute': safe_globals['execute']
        }

    except SecurityError as e:
        logging.warning(f"[Library Learning] Security error: {e}")
        return None
    except Exception as e:
        logging.warning(f"[Library Learning] Compilation error: {e}")
        return None

def evaluate_operator(
    operator: Dict,
    holdout_eps: List[Dict],
    train_eps: List[Dict],
    oracle: Optional[Callable]
) -> float:
    """Evaluate operator on holdout + perturbations.

    Score = weighted combination of:
    - Holdout accuracy (most important)
    - Perturbation robustness
    - No false positives on train

    Returns score in [0, 1]
    """

    trigger_fn = operator['trigger']
    execute_fn = operator['execute']

    holdout_correct = 0
    holdout_total = 0

    for ep in holdout_eps:
        query = ep['query']
        expected = ep.get('solution', ep.get('answer', ''))

        try:
            if not trigger_fn(query):
                continue

            holdout_total += 1
            result = execute_fn(query)

            if oracle:
                ok, _ = oracle(query, str(result))
                if ok:
                    holdout_correct += 1
            else:

                if str(result).strip() == str(expected).strip():
                    holdout_correct += 1

        except Exception as e:
            holdout_total += 1
            logging.debug(f"[Library Learning] Execution error: {e}")

    holdout_acc = holdout_correct / holdout_total if holdout_total > 0 else 0.0

    perturb_correct = 0
    perturb_total = 0

    for ep in holdout_eps[:2]:
        perturbations = generate_perturbations(ep['query'])

        for perturbed_query in perturbations:
            try:
                if not trigger_fn(perturbed_query):
                    continue

                perturb_total += 1
                result = execute_fn(perturbed_query)

                if result and not str(result).startswith("Error"):
                    perturb_correct += 1

            except Exception as e:
                logging.warning(f"[Learning] Perturbation test failed: {e}")
                perturb_total += 1

    perturb_acc = perturb_correct / perturb_total if perturb_total > 0 else 0.5

    train_trigger = 0
    for ep in train_eps:
        try:
            if trigger_fn(ep['query']):
                train_trigger += 1
        except Exception as e:
            logging.warning(f"[Learning] Trigger function failed: {e}")

    train_coverage = train_trigger / len(train_eps) if train_eps else 0.0

    score = (
        0.7 * holdout_acc +
        0.2 * perturb_acc +
        0.1 * train_coverage
    )

    logging.debug(f"[Library Learning] Eval: holdout={holdout_acc:.2f}, perturb={perturb_acc:.2f}, coverage={train_coverage:.2f}")

    return score

def generate_perturbations(query: str, n: int = 3) -> List[str]:
    """Generate simple perturbations of query for robustness testing.

    Examples:
    - "Calculate sum of 5 and 3" → "Calculate sum of 10 and 7"
    - "What is 2 + 2?" → "What is 5 + 3?"
    """

    import re

    perturbations = []

    numbers = re.findall(r'\d+', query)
    if numbers:
        for _ in range(n):
            perturbed = query
            for num in numbers:

                new_num = str(int(num) + np.random.randint(1, 10))
                perturbed = perturbed.replace(num, new_num, 1)
            perturbations.append(perturbed)

    return perturbations[:n]
