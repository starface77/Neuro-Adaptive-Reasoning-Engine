"""
Meta-Abduction Engine: Cross-domain meta-rule discovery.

The highest evolution level of NARE. Analyzes the skill registry
to discover structural isomorphisms between different problem domains,
generating cross-domain meta-rules that abstract over specific skills.

Example: If the system has separate skills for:
  - "Find shortest path in a graph"
  - "Optimize delivery route logistics"
  - "Find minimum cost network flow"
It should discover the meta-rule: "Graph optimization via shortest-path algorithms"
that unifies all three into a single, more abstract pattern.
"""

import ast
import collections
import json
import os
import re
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST fingerprinting (Tier A3): structural signatures over compiled skills.
# ---------------------------------------------------------------------------
#
# Replaces the legacy 9-boolean ``has_regex / has_loop / has_math`` Jaccard
# with a real structural signature: a *multiset* of AST node types appearing
# in the body of the skill's primary function (``solve`` if present, else
# ``execute``). Two skills are clustered together when the multiset Jaccard
# of their fingerprints exceeds a threshold — this is the cheap, well-known
# proxy for structural isomorphism (full anti-unification / e-graph
# matching is out of scope for this prototype).
#
# Why this matters for the paper alignment: the legacy keyword-Jaccard
# happily merges two skills that share ``for``/``if`` even when they
# compute completely unrelated things. The AST signature counts e.g.
# ``BinOp.Add`` separately from ``BinOp.Sub``, so "x = a + b in a loop"
# does NOT collapse with "x = max(a, b) in a loop".


# Nodes whose presence carries no structural information (we ignore them
# so two skills don't get spuriously similar through scaffolding).
_AST_IGNORE_NODES = {
    "Module",
    "Load",
    "Store",
    "Del",
    "Param",
    "FunctionDef",
    "arguments",
    "arg",
    "Expression",
    "Index",
    "alias",
    "ImportFrom",
    "Import",
}

# For BinOp / Compare / UnaryOp / BoolOp we want the operator type to
# count too, so "a + b" fingerprints differently from "a - b".
_OP_BEARING_NODES = {"BinOp", "UnaryOp", "Compare", "BoolOp", "AugAssign"}


def _primary_function_body(tree: ast.AST) -> Optional[List[ast.AST]]:
    """Return the body of ``solve`` if defined, else ``execute``, else None."""
    solve_body = None
    execute_body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "solve" and solve_body is None:
                solve_body = node.body
            elif node.name == "execute" and execute_body is None:
                execute_body = node.body
    return solve_body or execute_body


def ast_fingerprint(python_code: str) -> Optional[Dict[str, int]]:
    """Compute a multiset of structural tokens for a skill's primary fn.

    Returns ``None`` if the code does not parse or has no
    ``solve``/``execute``. Returns a dict mapping token -> count
    otherwise. Tokens are AST node names ("If", "For", "Call", "Return"
    etc.) plus operator-tagged variants for op-bearing nodes
    ("BinOp.Add", "Compare.Lt", "BoolOp.Or", ...).
    """
    if not python_code or not isinstance(python_code, str):
        return None
    try:
        tree = ast.parse(python_code)
    except SyntaxError:
        return None

    body = _primary_function_body(tree)
    if body is None:
        return None

    counts: Dict[str, int] = collections.Counter()
    for stmt in body:
        for node in ast.walk(stmt):
            name = type(node).__name__
            if name in _AST_IGNORE_NODES:
                continue
            counts[name] += 1
            if name in _OP_BEARING_NODES:
                op = getattr(node, "op", None)
                if op is not None:
                    counts[f"{name}.{type(op).__name__}"] += 1
                ops = getattr(node, "ops", None)
                if ops:
                    for o in ops:
                        counts[f"{name}.{type(o).__name__}"] += 1

    return dict(counts)


def ast_jaccard(a: Optional[Dict[str, int]], b: Optional[Dict[str, int]]) -> float:
    """Multiset Jaccard between two AST fingerprints.

    Defined as ``sum(min(a[k], b[k])) / sum(max(a[k], b[k]))`` over the
    union of keys. Returns 0.0 if either argument is None or empty.
    """
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    if union == 0:
        return 0.0
    return inter / union


class MetaAbductionEngine:
    """Discovers cross-domain meta-rules from existing skills."""

    def __init__(self, persist_dir: str = "memory_store"):
        self.persist_dir = persist_dir
        self.meta_rules: List[Dict[str, Any]] = []
        self._load()

    def analyze_skills(self, semantic_rules: List[Dict[str, Any]],
                       episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze existing skills to find structural isomorphisms.
        
        Groups skills by structural similarity of their code patterns
        and abstract signatures, then generates meta-rules.
        """
        if len(semantic_rules) < 2:
            return []

        # Step 1: Extract structural features from each skill
        skill_features = []
        for rule in semantic_rules:
            features = self._extract_structural_features(rule)
            if features:
                skill_features.append((rule, features))

        if len(skill_features) < 2:
            return []

        # Step 2: Cluster skills by structural similarity
        clusters = self._cluster_by_structure(skill_features)

        # Step 3: Generate meta-rules for each cluster
        new_meta_rules = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            meta_rule = self._generate_meta_rule(cluster, episodes)
            if meta_rule:
                new_meta_rules.append(meta_rule)

        # Step 4: Merge with existing meta-rules
        for new_mr in new_meta_rules:
            if not self._is_duplicate(new_mr):
                self.meta_rules.append(new_mr)
                logger.info(f"[MetaAbduction] New meta-rule: '{new_mr['name']}' "
                             f"(covers {len(new_mr['source_skills'])} skills)")

        self._save()
        return new_meta_rules

    def get_applicable_meta_rules(self, query: str) -> List[Dict[str, Any]]:
        """Find meta-rules that might apply to a given query."""
        applicable = []
        query_lower = query.lower()
        for mr in self.meta_rules:
            keywords = mr.get('keywords', [])
            if any(kw.lower() in query_lower for kw in keywords):
                applicable.append(mr)
            # Also check abstract pattern match
            pattern = mr.get('abstract_pattern', '').lower()
            if pattern and any(w in query_lower for w in pattern.split()[:5]):
                if mr not in applicable:
                    applicable.append(mr)
        return applicable

    def _extract_structural_features(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structural features from a skill's code for comparison.

        Tier A3: the primary signal is now the AST fingerprint of the
        skill's ``solve``/``execute`` body — a multiset of AST node
        types (with operator tags) — rather than a handful of
        ``"for"`` / ``"if"`` substring booleans. The legacy keyword
        booleans are kept as auxiliary metadata so the meta-rule
        generation step can still produce human-readable descriptions
        ("involves loops + math").
        """
        code = rule.get('python_code', '')
        if not code:
            return None

        fingerprint = ast_fingerprint(code)
        if fingerprint is None:
            # Skill that doesn't parse or has no solve/execute is not
            # eligible for structural meta-abduction.
            return None

        features = {
            'pattern': rule.get('pattern', ''),
            'ast_fingerprint': fingerprint,
            # Auxiliary keyword flags (kept for human-readable meta-rule
            # generation only; NOT used in clustering).
            'has_regex': 're.' in code or 'regex' in code.lower(),
            'has_math': 'math.' in code or any(op in code for op in ['+', '-', '*', '/', '**']),
            'has_loop': 'for ' in code or 'while ' in code,
            'has_recursion': code.count('def ') > 2,
            'has_string_ops': '.split(' in code or '.strip(' in code or '.replace(' in code,
            'has_list_ops': '.append(' in code or '.sort(' in code or 'sorted(' in code,
            'has_dict_ops': '.get(' in code or '.items(' in code or '.keys(' in code,
            'has_conditionals': 'if ' in code,
            'num_functions': code.count('def '),
            'code_length': len(code),
            'uses_numbers': bool(re.search(r'\b\d+\b', code)),
            'abstract_class': self._classify_domain(rule),
        }
        return features

    def _classify_domain(self, rule: Dict[str, Any]) -> str:
        """Classify a skill into a broad domain category."""
        pattern = rule.get('pattern', '').lower()
        code = rule.get('python_code', '').lower()
        combined = pattern + ' ' + code

        domain_keywords = {
            'mathematical': ['sum', 'product', 'factorial', 'fibonacci', 'prime',
                           'sequence', 'formula', 'calculate', 'compute', 'arithmetic'],
            'text_processing': ['parse', 'extract', 'split', 'regex', 'email',
                              'ip', 'address', 'text', 'string', 'word'],
            'data_structure': ['sort', 'search', 'tree', 'graph', 'list',
                             'array', 'stack', 'queue', 'hash'],
            'optimization': ['max', 'min', 'optimal', 'shortest', 'longest',
                           'efficient', 'cost', 'profit', 'dynamic'],
            'logic': ['boolean', 'true', 'false', 'and', 'or', 'not',
                     'condition', 'valid', 'check', 'verify'],
        }

        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)
        return 'general'

    # Threshold for AST-fingerprint Jaccard above which two skills are
    # considered structurally isomorphic enough to share a meta-rule.
    # 0.55 was chosen so that two arithmetic skills with the same
    # operator mix cluster, but a string-processing skill that happens
    # to share Call/Return nodes does not.
    AST_CLUSTER_THRESHOLD: float = 0.55

    def _cluster_by_structure(
        self,
        skill_features: List[Tuple[Dict, Dict]],
    ) -> List[List[Tuple[Dict, Dict]]]:
        """Cluster skills by AST-fingerprint Jaccard similarity.

        Tier A3: replaces the legacy 9-boolean Jaccard. Two skills end
        up in the same cluster when the multiset Jaccard of their AST
        fingerprints (in the body of ``solve``/``execute``) is at
        least :pyattr:`AST_CLUSTER_THRESHOLD`. Domain agreement adds
        a small bonus but is no longer the dominant signal — domain is
        a string label that only exists at meta-rule annotation time.
        """
        if not skill_features:
            return []

        clusters: List[List[Tuple[Dict, Dict]]] = [[sf] for sf in skill_features]

        def structural_similarity(f1: Dict, f2: Dict) -> float:
            fp1 = f1.get('ast_fingerprint')
            fp2 = f2.get('ast_fingerprint')
            sim = ast_jaccard(fp1, fp2)
            # Same-domain bonus capped at +0.05 so it can break ties
            # but cannot rescue structurally dissimilar skills.
            if (
                f1.get('abstract_class')
                and f1.get('abstract_class') == f2.get('abstract_class')
            ):
                sim = min(1.0, sim + 0.05)
            return sim

        changed = True
        while changed:
            changed = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if not clusters[i] or not clusters[j]:
                        continue
                    sim = structural_similarity(
                        clusters[i][0][1], clusters[j][0][1]
                    )
                    if sim >= self.AST_CLUSTER_THRESHOLD:
                        clusters[i].extend(clusters[j])
                        clusters[j] = []
                        changed = True
            clusters = [c for c in clusters if c]

        return clusters

    def _generate_meta_rule(self, cluster: List[Tuple[Dict, Dict]],
                             episodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a meta-rule from a cluster of structurally similar skills.
        
        Uses LLM to formulate abstract epistemological principles from the
        cluster of skills, discovering structural isomorphisms across domains.
        Falls back to rule-based generation if LLM is unavailable.
        """
        if len(cluster) < 2:
            return None

        rules = [r for r, _ in cluster]
        features = [f for _, f in cluster]

        # Find common patterns
        patterns = [r.get('pattern', '') for r in rules]
        domains = [f.get('abstract_class', 'general') for f in features]
        primary_domain = max(set(domains), key=domains.count)

        # Extract common structural operations
        common_ops = []
        op_keys = {
            'has_regex': 'pattern matching',
            'has_math': 'mathematical computation',
            'has_loop': 'iterative processing',
            'has_string_ops': 'string manipulation',
            'has_list_ops': 'collection operations',
        }
        for key, desc in op_keys.items():
            if all(f.get(key, False) for f in features):
                common_ops.append(desc)

        # Generate keywords from patterns
        keywords = set()
        for p in patterns:
            words = re.findall(r'[a-zA-Z]+', p.lower())
            keywords.update(w for w in words if len(w) > 3)

        # --- LLM-based abstract principle generation ---
        abstract_principle = ""
        llm_keywords = []
        try:
            abstract_principle, llm_keywords = self._llm_generate_principle(rules)
            if llm_keywords:
                keywords.update(llm_keywords)
        except Exception as e:
            logger.warning(f"[MetaAbduction] LLM principle generation failed: {e}")

        meta_rule = {
            'name': f"Meta-{primary_domain.title()}: {', '.join(patterns[:3])}",
            'domain': primary_domain,
            'abstract_pattern': abstract_principle or (
                f"Problems involving {primary_domain} with "
                f"{', '.join(common_ops[:3]) if common_ops else 'mixed operations'}"
            ),
            'source_skills': [r.get('pattern', 'Unknown') for r in rules],
            'common_operations': common_ops,
            'keywords': list(keywords)[:15],
            'num_source_skills': len(rules),
            'confidence': float(np.mean([r.get('confidence', 0.5) for r in rules])),
        }

        return meta_rule

    def _llm_generate_principle(self, rules: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Use LLM to discover abstract epistemological principles from skills.
        
        Returns (abstract_principle, keywords).
        """
        try:
            from nare.llm import _ensure_api_key, _post, API_KEY
            _ensure_api_key()
        except Exception:
            return "", []

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={API_KEY}"

        skills_desc = ""
        for r in rules[:5]:
            skills_desc += f"Skill: {r.get('pattern', 'Unknown')}\n"
            code = r.get('python_code', '')[:400]
            if code:
                skills_desc += f"Code:\n{code}\n"
            skills_desc += "---\n"

        prompt = f"""You are a meta-cognition engine performing STRUCTURAL ISOMORPHISM ANALYSIS.

Below are {len(rules)} compiled skills that share structural similarities. Your task is to discover the ABSTRACT EPISTEMOLOGICAL PRINCIPLE that unifies them — the deep structural pattern common to all.

SKILLS:
{skills_desc}

INSTRUCTIONS:
1. Identify the structural isomorphism: What abstract algorithm or reasoning pattern do ALL these skills share?
2. Formulate a GENERAL PRINCIPLE that could apply to entirely new domains where the same structural pattern appears.
3. Extract 5-10 KEYWORDS that would help identify new problems matching this meta-pattern.

Output in EXACTLY this format:
PRINCIPLE: [One paragraph describing the abstract meta-rule — the deep structural pattern shared by all skills, stated in domain-independent terms]
KEYWORDS: [comma-separated keywords that would trigger this meta-rule for new, unseen problems]"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4}
        }

        try:
            res = _post(url, payload)
            content = res.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

            principle = ""
            llm_keywords = []

            principle_match = re.search(r'PRINCIPLE:\s*(.*?)(?=KEYWORDS:|$)', content, re.DOTALL)
            if principle_match:
                principle = principle_match.group(1).strip()[:500]

            keywords_match = re.search(r'KEYWORDS:\s*(.*)', content)
            if keywords_match:
                raw = keywords_match.group(1).strip()
                llm_keywords = [w.strip().lower() for w in raw.split(',') if len(w.strip()) > 2][:10]

            logger.info(f"[MetaAbduction] LLM principle: {principle[:100]}...")
            return principle, llm_keywords
        except Exception as e:
            logger.warning(f"[MetaAbduction] LLM call failed: {e}")
            return "", []

    def _is_duplicate(self, new_mr: Dict[str, Any]) -> bool:
        """Check if a meta-rule already exists."""
        for existing in self.meta_rules:
            if existing['name'] == new_mr['name']:
                return True
            # Check source skill overlap
            overlap = set(existing['source_skills']) & set(new_mr['source_skills'])
            if len(overlap) > len(new_mr['source_skills']) * 0.5:
                return True
        return False

    def _save(self):
        path = os.path.join(self.persist_dir, "meta_rules.json")
        try:
            serializable = []
            for mr in self.meta_rules:
                s = dict(mr)
                if 'confidence' in s:
                    s['confidence'] = float(s['confidence'])
                serializable.append(s)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save meta-rules: {e}")

    def _load(self):
        path = os.path.join(self.persist_dir, "meta_rules.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.meta_rules = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load meta-rules: {e}")
