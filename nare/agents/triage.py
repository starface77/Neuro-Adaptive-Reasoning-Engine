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
import logging
from typing import Literal

from nare.reasoning import llm

IntentType = Literal["QUESTION", "EXPLORE", "EDIT"]

# Fast heuristic patterns — avoid LLM call for obvious cases
# NOTE: Use lookahead/lookbehind for Cyrillic to handle word boundaries properly
_EDIT_PATTERNS = re.compile(
    r'(?:^|[^а-яА-Яa-zA-Z])(fix|add|implement|create|refactor|change|update|modify|remove|delete|rename|move|replace|write|build|make|изучай|изучпай|посмотри|проверь|доработай|улучши|сделай|делай|добавь|почини|исправь|напиши|пиши|создай|удали|измени|реализуй)(?:[^а-яА-Яa-zA-Z]|$)',
    re.IGNORECASE,
)
_QUESTION_PATTERNS = re.compile(
    r'(?:^|[^а-яА-Яa-zA-Z])(what|why|how|where|when|explain|describe|show|tell|is there|does it|can you|hello|hi|hey|привет|ку|как|зачем|почему|что|где|объясни|расскажи|помоги|работает|здравствуй|йо|sup|оцени|покажи|analyze|evaluate|assess|review|check)(?:[^а-яА-Яa-zA-Z]|$)',
    re.IGNORECASE,
)

# Short greetings/questions (1-3 chars) - always QUESTION
_SHORT_GREETINGS = re.compile(r'^(hi|ку|йо|ok|да|нет|yes|no)$', re.IGNORECASE)


class TriageAgent:
    """Classify user intent in <1s using heuristics + optional LLM fallback."""

    def __init__(self):
        self.logger = logging.getLogger("nare.agents.triage")

    def classify(self, query: str, use_llm_fallback: bool = False) -> IntentType:
        """Classify the query intent.

        Heuristic-first: pattern matching covers ~90% of cases.
        LLM fallback for ambiguous queries (costs ~50 tokens).
        """
        q = query.strip()

        # ── Short greetings check first ──────────────────
        if _SHORT_GREETINGS.match(q):
            self.logger.info("[Triage] QUESTION (short greeting)")
            return "QUESTION"

        # ── Heuristic pass ───────────────────────────────
        has_edit = bool(_EDIT_PATTERNS.search(q))
        has_question = bool(_QUESTION_PATTERNS.search(q))

        self.logger.info(f"[Triage] Query: {q[:50]}...")
        self.logger.info(f"[Triage] has_edit={has_edit}, has_question={has_question}")

        # If it ends with '?' it's almost certainly a question
        if q.endswith('?') and not has_edit:
            self.logger.info("[Triage] QUESTION (ends with ?)")
            return "QUESTION"

        # Strong edit signals - PRIORITIZE EDIT over QUESTION
        if has_edit:
            self.logger.info("[Triage] EDIT (has edit keyword)")
            return "EDIT"

        # Strong question signals
        if has_question:
            self.logger.info("[Triage] QUESTION (heuristic)")
            return "QUESTION"

        # ── Ambiguous — use LLM if enabled ───────────────
        if use_llm_fallback:
            return self._llm_classify(q)

        # Default: treat ambiguous as EDIT (safer — plan before acting)
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
            if samples:
                raw = samples[0]["solution"].strip().upper()
                for intent in ("QUESTION", "EXPLORE", "EDIT"):
                    if intent in raw:
                        self.logger.info(f"[Triage] {intent} (LLM)")
                        return intent
        except Exception as e:
            self.logger.warning(f"[Triage] LLM fallback failed: {e}")

        return "EDIT"
