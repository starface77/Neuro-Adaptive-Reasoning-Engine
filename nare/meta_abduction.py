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

import json
import os
import re
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
        """Extract structural features from a skill's code for comparison."""
        code = rule.get('python_code', '')
        if not code:
            return None

        features = {
            'pattern': rule.get('pattern', ''),
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

    def _cluster_by_structure(self, 
                               skill_features: List[Tuple[Dict, Dict]]) -> List[List[Tuple[Dict, Dict]]]:
        """Cluster skills by structural feature similarity."""
        if not skill_features:
            return []

        # Simple agglomerative clustering based on feature overlap
        clusters = [[sf] for sf in skill_features]
        
        def feature_similarity(f1: Dict, f2: Dict) -> float:
            """Compute Jaccard-like similarity between feature sets."""
            bool_keys = ['has_regex', 'has_math', 'has_loop', 'has_recursion',
                        'has_string_ops', 'has_list_ops', 'has_dict_ops',
                        'has_conditionals', 'uses_numbers']
            matches = sum(1 for k in bool_keys if f1.get(k) == f2.get(k))
            total = len(bool_keys)
            
            # Bonus for same domain
            if f1.get('abstract_class') == f2.get('abstract_class'):
                matches += 2
                total += 2
            else:
                total += 2
            
            return matches / total if total > 0 else 0.0

        # Merge clusters with similarity > 0.7
        changed = True
        while changed:
            changed = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if not clusters[i] or not clusters[j]:
                        continue
                    # Compare representative elements
                    sim = feature_similarity(
                        clusters[i][0][1], clusters[j][0][1]
                    )
                    if sim >= 0.7:
                        clusters[i].extend(clusters[j])
                        clusters[j] = []
                        changed = True
            clusters = [c for c in clusters if c]

        return clusters

    def _generate_meta_rule(self, cluster: List[Tuple[Dict, Dict]],
                             episodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a meta-rule from a cluster of structurally similar skills."""
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

        meta_rule = {
            'name': f"Meta-{primary_domain.title()}: {', '.join(patterns[:3])}",
            'domain': primary_domain,
            'abstract_pattern': f"Problems involving {primary_domain} with "
                               f"{', '.join(common_ops[:3]) if common_ops else 'mixed operations'}",
            'source_skills': [r.get('pattern', 'Unknown') for r in rules],
            'common_operations': common_ops,
            'keywords': list(keywords)[:10],
            'num_source_skills': len(rules),
            'confidence': np.mean([r.get('confidence', 0.5) for r in rules]),
        }

        return meta_rule

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
