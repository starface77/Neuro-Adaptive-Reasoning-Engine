import logging
from typing import List, Dict, Any, Optional, Callable

class Critic:
    def evaluate(
        self,
        query: str,
        candidates: List[Dict],
        oracle: Optional[Callable] = None
    ) -> List[Dict]:
        if not candidates:
            return []

        for c in candidates:
            if oracle:
                try:
                    passed, info = oracle(query, c['solution'])
                    c['final_score'] = 1.0 if passed else 0.0
                except Exception as e:
                    logging.warning(f"Oracle failed: {e}")
                    c['final_score'] = 0.0
            else:
                c['final_score'] = 0.5

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates
