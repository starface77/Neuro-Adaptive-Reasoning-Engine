"""
Triage Agent — Intent Classification

Fast, cheap classifier that determines the type of user request
before engaging the expensive planning/coding pipeline.

Intent types:
  QUESTION — Explain code, answer about architecture. No edits needed.
  EXPLORE  — Search for bugs, analyze logs, investigate issues.
  EDIT     — Modify codebase: fix bug, add feature, refactor.
"""

import re
from nare.utils.logger import get_logger
from typing import Literal

from nare.reasoning import llm

IntentType = Literal["QUESTION", "EXPLORE", "EDIT"]

_EDIT_PATTERNS = re.compile(
    r'(?:^|[^а-яА-Яa-zA-Z])(fix|add|implement|create|refactor|change|update|modify|remove|delete|rename|move|replace|write|build|make|изучай|изучпай|посмотри|проверь|доработай|улучши|сделай|делай|добавь|почини|исправь|напиши|пиши|создай|удали|измени|реализуй)(?:[^а-яА-Яa-zA-Z]|$)',
    re.IGNORECASE,
)
_QUESTION_PATTERNS = re.compile(
    r'(?:^|[^а-яА-Яa-zA-Z])(what|why|how|where|when|explain|describe|show|tell|is there|does it|can you|hello|hi|hey|привет|ку|как|зачем|почему|что|где|объясни|расскажи|помоги|работает|здравствуй|йо|sup|оцени|покажи|analyze|evaluate|assess|review|check)(?:[^а-яА-Яa-zA-Z]|$)',
    re.IGNORECASE,
)

_SHORT_GREETINGS = re.compile(r'^(hi|ку|йо|ok|да|нет|yes|no)$', re.IGNORECASE)

class TriageAgent:
    """Classify user intent in <1s using heuristics + optional LLM fallback."""

    def __init__(self):
        self.logger = get_logger("nare.agents.roles.triage")

    def classify(self, query: str, use_llm_fallback: bool = False) -> IntentType:
        """Classify the query intent.

        Heuristic-first: pattern matching covers ~90% of cases.
        LLM fallback for ambiguous queries (costs ~50 tokens).
        """
        q = query.strip()

        if _SHORT_GREETINGS.match(q):
            self.logger.info("[Triage] QUESTION (short greeting)")
            return "QUESTION"

        has_edit = bool(_EDIT_PATTERNS.search(q))
        has_question = bool(_QUESTION_PATTERNS.search(q))

        self.logger.info(f"[Triage] Query: {q[:50]}...")
        self.logger.info(f"[Triage] has_edit={has_edit}, has_question={has_question}")

        if q.endswith('?') and not has_edit:
            self.logger.info("[Triage] QUESTION (ends with ?)")
            return "QUESTION"

        if has_edit:
            self.logger.info("[Triage] EDIT (has edit keyword)")
            return "EDIT"

        if has_question:
            self.logger.info("[Triage] QUESTION (heuristic)")
            return "QUESTION"

        if use_llm_fallback:
            return self._llm_classify(q)

        self.logger.info("[Triage] EDIT (default for ambiguous)")
        return "EDIT"

    def _llm_classify(self, query: str) -> IntentType:
        """Use LLM for ambiguous cases. Very cheap (~50 tokens)."""
        prompt = (
            f"Classify the following user request into exactly one category.\n"
            f"Categories: QUESTION, EXPLORE, EDIT\n\n"
            f"Request: {query}\n\n"
            f"Output ONLY one word: QUESTION, EXPLORE, or EDIT."
        )
        try:
            samples, _ = llm.generate_samples(prompt, n=1, temperature=0.0, mode="DIRECT")
            if samples and len(samples) > 0 and isinstance(samples[0], dict) and "solution" in samples[0]:
                raw = samples[0]["solution"].strip().upper()
                for intent in ("QUESTION", "EXPLORE", "EDIT"):
                    if intent in raw:
                        self.logger.info(f"[Triage] {intent} (LLM)")
                        return intent
        except Exception as e:
            self.logger.warning(f"[Triage] LLM fallback failed: {e}")

        return "EDIT"
