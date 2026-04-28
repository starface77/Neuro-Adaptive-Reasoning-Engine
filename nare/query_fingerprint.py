"""Deterministic structural fingerprint for natural-language queries.

Motivation
----------

Sleep-time clustering originally relied **only** on dense Gemini embedding
neighbourhoods of episode signatures. For 3072-dim embeddings, two
queries that share surface vocabulary can have cosine ~0.7 even when
their underlying *task structure* is unrelated (and conversely, two
queries with the same arithmetic structure but different surface
vocabulary — "У Пети 5 яблок, отдал 2" vs. "У NASA 5 шаттлов, 2
сгорели" — can score below the cluster threshold despite being
literally the same `5 - 2` problem).

This module computes a complementary *structural fingerprint* of the
query as a multiset of token-class features that are invariant to
surface noun substitution but sensitive to operation type, arity, and
question intent. It is used as a **secondary gate** during sleep
cluster trigger and consolidation: a candidate cluster only fires if
the embedding-density signal AND the multiset Jaccard over query
fingerprints both exceed their respective thresholds.

The fingerprint is intentionally crude (regex + keyword lists). It is
NOT a parser. It is a deterministic, fast, language-agnostic-ish
signal that complements (not replaces) embeddings.
"""

from __future__ import annotations

import re
import collections
from typing import Dict, Iterable, Optional


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

# Operation keywords. Russian + English. Multi-word phrases checked first.
_OPERATION_PHRASES: Dict[str, str] = {
    # English
    "sum of": "OP.SUM",
    "total of": "OP.SUM",
    "added to": "OP.ADD",
    "added together": "OP.ADD",
    "subtracted from": "OP.SUB",
    "difference between": "OP.SUB",
    "multiplied by": "OP.MUL",
    "divided by": "OP.DIV",
    "ratio of": "OP.DIV",
    "average of": "OP.AVG",
    "mean of": "OP.AVG",
    "percent of": "OP.PCT",
    "percentage of": "OP.PCT",
    "how many": "Q.COUNT",
    "how much": "Q.AMOUNT",
    "list of": "Q.LIST",
    "extract from": "Q.EXTRACT",
    "find all": "Q.LIST",
    # Russian
    "сколько всего": "OP.SUM",
    "сумма": "OP.SUM",
    "сложить": "OP.ADD",
    "прибавить": "OP.ADD",
    "вычесть": "OP.SUB",
    "разница": "OP.SUB",
    "отдал": "OP.SUB",
    "осталось": "OP.SUB",
    "умножить": "OP.MUL",
    "разделить": "OP.DIV",
    "среднее": "OP.AVG",
    "процент": "OP.PCT",
    "сколько": "Q.COUNT",
    "найди": "Q.EXTRACT",
    "извлеки": "Q.EXTRACT",
    "перечисли": "Q.LIST",
    "выпиши": "Q.LIST",
}

_OPERATION_TOKENS: Dict[str, str] = {
    # English
    "add": "OP.ADD",
    "plus": "OP.ADD",
    "sum": "OP.SUM",
    "total": "OP.SUM",
    "subtract": "OP.SUB",
    "minus": "OP.SUB",
    "less": "OP.SUB",
    "multiply": "OP.MUL",
    "times": "OP.MUL",
    "product": "OP.MUL",
    "divide": "OP.DIV",
    "quotient": "OP.DIV",
    "average": "OP.AVG",
    "mean": "OP.AVG",
    "median": "OP.MED",
    "min": "OP.MIN",
    "max": "OP.MAX",
    "count": "Q.COUNT",
    "extract": "Q.EXTRACT",
    "parse": "Q.EXTRACT",
    "list": "Q.LIST",
    "compare": "Q.COMPARE",
    "translate": "Q.TRANSLATE",
    "summarize": "Q.SUMMARIZE",
    "classify": "Q.CLASSIFY",
    "explain": "Q.EXPLAIN",
    # Russian
    "сложение": "OP.ADD",
    "вычитание": "OP.SUB",
    "умножение": "OP.MUL",
    "деление": "OP.DIV",
    "сравни": "Q.COMPARE",
    "переведи": "Q.TRANSLATE",
    "объясни": "Q.EXPLAIN",
    "классифицируй": "Q.CLASSIFY",
}


def _bin_number_count(n: int) -> str:
    if n == 0:
        return "NUMS.0"
    if n == 1:
        return "NUMS.1"
    if n == 2:
        return "NUMS.2"
    if n <= 5:
        return "NUMS.SMALL"  # 3..5
    return "NUMS.MANY"        # 6+


def query_fingerprint(query: str) -> Dict[str, int]:
    """Return a multiset (Counter as dict) of structural feature tags.

    Tags include:

    * ``NUMS.{0,1,2,SMALL,MANY}`` — coarse count of numeric literals
    * ``OP.{ADD,SUB,MUL,DIV,SUM,AVG,...}`` — arithmetic operation hints
    * ``Q.{COUNT,LIST,EXTRACT,COMPARE,TRANSLATE,...}`` — question intent
    * ``HAS.{QUESTION,CURRENCY,PERCENT,DATE_ISO,EMAIL,URL,CODE_FENCE}``
      — surface markers
    * ``LEN.{SHORT,MED,LONG}`` — query length bucket

    All tags are language-agnostic identifiers; only the keyword tables
    are language-specific.
    """
    if not isinstance(query, str) or not query.strip():
        return {}

    text = query.strip()
    text_lower = text.lower()
    fp: collections.Counter = collections.Counter()

    # 1. Numeric literals.
    numbers = _NUMBER_RE.findall(text)
    fp[_bin_number_count(len(numbers))] += 1
    for raw in numbers:
        normalized = raw.replace(",", ".")
        try:
            value = float(normalized)
        except ValueError:
            continue
        if value == 0:
            fp["NUM.ZERO"] += 1
        elif abs(value) < 1:
            fp["NUM.FRAC"] += 1
        elif abs(value) < 100:
            fp["NUM.SMALL"] += 1
        elif abs(value) < 10000:
            fp["NUM.MED"] += 1
        else:
            fp["NUM.LARGE"] += 1

    # 2. Multi-word operation phrases (highest specificity first).
    masked = text_lower
    for phrase, tag in _OPERATION_PHRASES.items():
        if phrase in masked:
            fp[tag] += masked.count(phrase)
            masked = masked.replace(phrase, " ")

    # 3. Single-token operation keywords (word-boundary).
    for token, tag in _OPERATION_TOKENS.items():
        # Crude word-boundary that handles ASCII + Cyrillic.
        pattern = re.compile(rf"(?:^|\W){re.escape(token)}(?:\W|$)", re.IGNORECASE)
        if pattern.search(masked):
            fp[tag] += 1

    # 4. Surface markers.
    if "?" in text:
        fp["HAS.QUESTION"] += 1
    if re.search(r"[$€£¥₽]", text):
        fp["HAS.CURRENCY"] += 1
    if re.search(r"\d+\s*%", text):
        fp["HAS.PERCENT"] += 1
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        fp["HAS.DATE_ISO"] += 1
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        fp["HAS.EMAIL"] += 1
    if re.search(r"https?://", text):
        fp["HAS.URL"] += 1
    if "```" in text:
        fp["HAS.CODE_FENCE"] += 1

    # 5. Length bucket.
    n_chars = len(text)
    if n_chars < 60:
        fp["LEN.SHORT"] += 1
    elif n_chars < 240:
        fp["LEN.MED"] += 1
    else:
        fp["LEN.LONG"] += 1

    return dict(fp)


def query_jaccard(a: Optional[Dict[str, int]], b: Optional[Dict[str, int]]) -> float:
    """Multiset Jaccard between two query fingerprints.

    Returns 0.0 if either side is empty / None. Returns 1.0 only when
    both fingerprints are non-empty and identical (multiset equality).
    """
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    if union == 0:
        return 0.0
    return inter / union


def average_pairwise_jaccard(fingerprints: Iterable[Optional[Dict[str, int]]]) -> float:
    """Average pairwise multiset Jaccard over a set of fingerprints.

    Useful for "is this candidate cluster structurally homogeneous?"
    gates. Returns 0.0 if fewer than two non-empty fingerprints are
    supplied.
    """
    fps = [fp for fp in fingerprints if fp]
    n = len(fps)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += query_jaccard(fps[i], fps[j])
            pairs += 1
    return total / pairs if pairs else 0.0


__all__ = [
    "query_fingerprint",
    "query_jaccard",
    "average_pairwise_jaccard",
]
