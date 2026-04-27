"""
Graph Memory Layer: Associative connections between episodic memories.

Implements graph-based retrieval on top of the flat FAISS index:
- Nodes = episodes (each episode is a node)
- Edges = semantic similarity + co-activation links
- Supports graph traversal for multi-hop retrieval
- Hebbian strengthening: edges used together grow stronger
"""

import json
import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EpisodeGraph:
    """Directed weighted graph over episodic memory indices."""

    def __init__(self, persist_dir: str = "memory_store"):
        self.persist_dir = persist_dir
        # adjacency: {node_id: {neighbor_id: weight, ...}}
        self.adjacency: Dict[int, Dict[int, float]] = {}
        # node metadata (cached for fast access)
        self.node_labels: Dict[int, str] = {}
        self._load()

    def add_node(self, node_id: int, label: str = ""):
        if node_id not in self.adjacency:
            self.adjacency[node_id] = {}
        if label:
            self.node_labels[node_id] = label

    def add_edge(self, src: int, dst: int, weight: float = 1.0):
        """Add or strengthen a directed edge."""
        if src not in self.adjacency:
            self.adjacency[src] = {}
        if dst not in self.adjacency:
            self.adjacency[dst] = {}
        old = self.adjacency[src].get(dst, 0.0)
        self.adjacency[src][dst] = min(1.0, old + weight)

    def strengthen_edge(self, src: int, dst: int, delta: float = 0.1):
        """Hebbian strengthening: co-activated episodes gain link weight."""
        if src in self.adjacency and dst in self.adjacency[src]:
            self.adjacency[src][dst] = min(1.0, self.adjacency[src][dst] + delta)
        else:
            self.add_edge(src, dst, weight=delta)

    def weaken_all(self, decay: float = 0.01):
        """Synaptic downscaling: all edges decay slightly."""
        to_remove = []
        for src in self.adjacency:
            for dst in list(self.adjacency[src]):
                self.adjacency[src][dst] -= decay
                if self.adjacency[src][dst] <= 0:
                    to_remove.append((src, dst))
        for s, d in to_remove:
            del self.adjacency[s][d]

    def neighbors(self, node_id: int, min_weight: float = 0.1) -> List[Tuple[int, float]]:
        """Get sorted neighbors by weight (descending)."""
        if node_id not in self.adjacency:
            return []
        edges = [(n, w) for n, w in self.adjacency[node_id].items() if w >= min_weight]
        return sorted(edges, key=lambda x: x[1], reverse=True)

    def multi_hop_retrieve(self, start_ids: List[int], hops: int = 2,
                           min_weight: float = 0.2) -> List[int]:
        """BFS graph traversal from start nodes, up to `hops` depth.
        
        Returns ordered list of reachable node IDs (excluding start nodes).
        """
        visited = set(start_ids)
        frontier = list(start_ids)
        results = []

        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                for neighbor, weight in self.neighbors(node, min_weight):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        results.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return results

    def build_from_episodes(self, episodes: List[Dict[str, Any]], 
                            similarity_threshold: float = 0.6):
        """Build graph edges from episode embeddings based on similarity.
        
        Creates edges between episodes whose embeddings are similar
        (above threshold), mimicking Hebbian co-activation.
        """
        import faiss

        embeddings = []
        valid_indices = []
        for i, ep in enumerate(episodes):
            if 'embedding' in ep:
                emb = np.array(ep['embedding'], dtype=np.float32)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                embeddings.append(emb)
                valid_indices.append(i)
                self.add_node(i, label=ep.get('query', '')[:50])

        if len(embeddings) < 2:
            return

        all_embs = np.vstack(embeddings)
        faiss.normalize_L2(all_embs)

        # Compute pairwise similarities
        inner_products = np.dot(all_embs, all_embs.T)

        for i_idx, i in enumerate(valid_indices):
            for j_idx, j in enumerate(valid_indices):
                if i >= j:
                    continue
                sim = float(inner_products[i_idx, j_idx])
                if sim >= similarity_threshold:
                    self.add_edge(i, j, weight=sim)
                    self.add_edge(j, i, weight=sim)

        logger.info(f"[Graph] Built graph: {len(valid_indices)} nodes, "
                     f"{sum(len(v) for v in self.adjacency.values())} edges")

    def save(self):
        path = os.path.join(self.persist_dir, "graph.json")
        data = {
            "adjacency": {str(k): {str(n): w for n, w in v.items()} 
                          for k, v in self.adjacency.items()},
            "labels": {str(k): v for k, v in self.node_labels.items()},
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save graph: {e}")

    def _load(self):
        path = os.path.join(self.persist_dir, "graph.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.adjacency = {
                int(k): {int(n): w for n, w in v.items()}
                for k, v in data.get("adjacency", {}).items()
            }
            self.node_labels = {int(k): v for k, v in data.get("labels", {}).items()}
        except Exception as e:
            logger.warning(f"Failed to load graph: {e}")
