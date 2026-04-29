import logging
from typing import List, Dict, Any, Optional, Tuple, Counter
from . import llm
from .config import DEFAULT_CONFIG, NareConfig

class HybridCritic:
    """Convergent selection critic.

    Combines four independent signals for robust candidate ranking:
      1. Ground Truth Validator — compilation / sandbox execution for
         code solutions. Mandatory when applicable.
      2. LLM-critic — pairwise Elo tournament as a fallback for "soft"
         cases without a deterministic verifier.
      3. Self-Consistency — majority-vote over extracted short answers.
      4. Rule-based heuristic — fast syntactic checks.
    """

    def __init__(self, config: NareConfig = DEFAULT_CONFIG):
        cfg = config.critic
        self.weights = (cfg.w_llm, cfg.w_rule, cfg.w_neural)
        self.elo_k = cfg.elo_k_factor
        self.elo_init = cfg.elo_initial_rating

    def _ground_truth_validate(self, solution: str) -> Optional[float]:
        from .sandbox import extract_python_block, validate_code
        code = extract_python_block(solution)
        if not code:
            return None

        try:
            validate_code(code)
            return 0.8
        except Exception:
            return 0.2

    def _rule_based_check(self, solution: str) -> float:
        score = 0.5
        if "```" in solution or "def " in solution:
            score += 0.3
        if "Error" in solution or "Exception" in solution:
            score -= 0.4
        return max(0.0, min(1.0, score))

    @staticmethod
    def _extract_short_answer(solution: str) -> str:
        lines = [ln.strip() for ln in solution.strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        answer = lines[-1]
        for prefix in ("Answer:", "Result:", "Output:", "answer:", "result:"):
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        answer = answer.strip("`").strip("*").strip()
        return answer.lower()

    def _self_consistency_scores(self, candidates: List[Dict]) -> Dict[int, float]:
        answers = [self._extract_short_answer(c['solution']) for c in candidates]
        if not answers:
            return {}
        counts = Counter(answers)
        total = len(answers)
        return {i: counts[a] / total for i, a in enumerate(answers)}

    def evaluate(
        self, query: str, candidates: List[Dict], oracle: Optional[Any] = None
    ) -> List[Dict]:
        if not candidates:
            return []
        
        # Single candidate case
        if len(candidates) == 1:
            c = candidates[0]
            c['llm_score'] = 0.5
            c['rule_score'] = self._rule_based_check(c['solution'])
            if oracle:
                try:
                    passed, _ = oracle(query, c['solution'])
                    gt = 1.0 if passed else 0.0
                except Exception:
                    gt = 0.0
            else:
                gt = self._ground_truth_validate(c['solution'])
            c['gt_score'] = gt
            c['sc_score'] = 1.0
            c['final_score'] = gt if gt is not None else 0.5
            return candidates

        # Multi-candidate case
        # 1. Ground Truth
        for c in candidates:
            if oracle:
                try:
                    passed, _ = oracle(query, c['solution'])
                    c['gt_score'] = 1.0 if passed else 0.0
                except Exception:
                    c['gt_score'] = 0.0
            else:
                c['gt_score'] = self._ground_truth_validate(c['solution'])

        # 2. Self-Consistency
        sc_scores = self._self_consistency_scores(candidates)
        for i, c in enumerate(candidates):
            c['sc_score'] = sc_scores.get(i, 0.0)

        # 3. Elo tournament
        for c in candidates:
            c['elo'] = self.elo_init

        for i in range(len(candidates) - 1):
            c1, c2 = candidates[i], candidates[i + 1]
            try:
                winner = llm.llm_pairwise_judge(query, c1['solution'], c2['solution'])
            except Exception as e:
                logging.error(f"Pairwise judge failed: {e}")
                winner = 1

            ea = 1.0 / (1.0 + 10.0 ** ((c2['elo'] - c1['elo']) / 400.0))
            if winner == 1:
                c1['elo'] += self.elo_k * (1.0 - ea)
                c2['elo'] -= self.elo_k * (1.0 - ea)
            else:
                c1['elo'] -= self.elo_k * ea
                c2['elo'] += self.elo_k * ea

        elos = [c['elo'] for c in candidates]
        e_min, e_max = min(elos), max(elos)

        # 4. Combine signals
        for c in candidates:
            c['llm_score'] = (c['elo'] - e_min) / (e_max - e_min) if e_max > e_min else 0.5
            c['rule_score'] = self._rule_based_check(c['solution'])

            if c['gt_score'] is not None:
                # GT has the highest weight
                c['final_score'] = max(0.0, 0.40 * c['gt_score'] + 0.25 * c['sc_score'] + 0.20 * c['llm_score'] + 0.15 * c['rule_score'])
            else:
                c['final_score'] = max(0.0, 0.30 * c['sc_score'] + self.weights[0] * 0.70 * c['llm_score'] + self.weights[1] * 0.70 * c['rule_score'])

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates
