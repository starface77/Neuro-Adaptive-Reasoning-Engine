"""
RL Retriever: Reinforcement learning for retrieval relevance maximization.

Uses a simple bandit-style RL approach to learn retrieval preferences:
- Each episode has a retrieval_value (learned via reward signal)
- When retrieved episodes lead to good outcomes, their value increases
- When retrieved episodes are irrelevant, their value decreases
- Over time, the retriever learns to prioritize high-value episodes

This avoids requiring PyTorch/TensorFlow by implementing a lightweight
contextual bandit with feature-based value estimation.
"""

import json
import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .config import DEFAULT_CONFIG, NareConfig

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid mapping R → (0, 1)."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


class RLRetriever:
    """Contextual bandit retriever that learns from outcome feedback.

    All knobs (lr, discount, exploration schedule, blend weights, weight
    clipping norm) live in :class:`nare.config.RetrievalConfig` rather
    than as ad-hoc magic numbers.
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        persist_dir: str = "memory_store",
        config: NareConfig = DEFAULT_CONFIG,
        learning_rate: Optional[float] = None,
        discount: Optional[float] = None,
    ):
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        self.config = config
        rcfg = config.retrieval

        self.lr = learning_rate if learning_rate is not None else rcfg.rl_lr
        self.discount = discount if discount is not None else rcfg.rl_discount

        # Value function: linear projection of embedding → scalar value
        self.weights = np.zeros(embedding_dim, dtype=np.float32)
        self.bias = 0.0

        # Per-episode retrieval rewards history
        self.episode_values: Dict[int, float] = {}

        self.epsilon = rcfg.rl_epsilon_initial
        self.total_updates = 0

        self._load()

    def predict_value(self, embedding: np.ndarray) -> float:
        """Predict the retrieval value of an episode given its embedding."""
        emb = np.array(embedding, dtype=np.float32).flatten()[:self.embedding_dim]
        return float(np.dot(self.weights, emb) + self.bias)

    def rerank(self, candidates: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Re-rank retrieved candidates using learned value function.
        
        Each candidate should have 'embedding' and 'similarity' fields.
        Returns candidates sorted by combined score (similarity + learned value).
        """
        if not candidates:
            return candidates

        rcfg = self.config.retrieval
        w_sim = rcfg.rerank_w_sim
        w_rl = rcfg.rerank_w_rl

        for cand in candidates:
            emb = cand.get('embedding')
            if emb is not None:
                rl_raw = self.predict_value(emb)
                # Squash unbounded RL value into [0,1] before blending
                # with cosine similarity — mixing two raw scales (sim is
                # bounded, predict_value is not) was unsound.
                rl_norm = _sigmoid(rl_raw)
                cand['rl_value'] = rl_raw
                cand['rl_value_normalized'] = rl_norm
                sim = cand.get('similarity', 0.5)
                cand['combined_score'] = w_sim * sim + w_rl * rl_norm
            else:
                cand['rl_value'] = 0.0
                cand['rl_value_normalized'] = 0.5
                cand['combined_score'] = cand.get('similarity', 0.5)

        # ε-greedy: with probability ε, sample softmax over scores rather
        # than full random shuffle (full shuffle threw away signal).
        if np.random.random() < self.epsilon and len(candidates) > 1:
            scores = np.array(
                [c['combined_score'] for c in candidates], dtype=np.float64
            )
            # Softmax with mild temperature.
            scores = scores - scores.max()
            probs = np.exp(scores / 0.5)
            probs = probs / probs.sum()
            order = np.random.choice(
                len(candidates), size=len(candidates), replace=False, p=probs
            )
            candidates = [candidates[i] for i in order]
            logger.info("[RL Retriever] Exploration: softmax-sampled order")
        else:
            candidates.sort(key=lambda c: c['combined_score'], reverse=True)

        if top_k:
            candidates = candidates[:top_k]

        return candidates

    def update(self, episode_id: int, embedding: np.ndarray, reward: float):
        """Update value function based on outcome reward.
        
        reward > 0: episode was useful (led to good answer)
        reward < 0: episode was irrelevant or harmful
        reward = 0: neutral
        """
        emb = np.array(embedding, dtype=np.float32).flatten()[:self.embedding_dim]

        # Current prediction
        predicted = self.predict_value(emb)
        
        # TD-style update: move prediction toward observed reward
        error = reward - predicted
        
        # Gradient update for linear model
        self.weights += self.lr * error * emb
        self.bias += self.lr * error
        
        # Clip weights to prevent divergence (norm ball).
        max_norm = self.config.retrieval.rl_weight_clip_norm
        norm = np.linalg.norm(self.weights)
        if norm > max_norm:
            self.weights *= max_norm / norm
        # Bias also clipped — previously unbounded.
        self.bias = float(np.clip(self.bias, -max_norm, max_norm))

        # Track per-episode values with exponential moving average
        old_val = self.episode_values.get(episode_id, 0.0)
        self.episode_values[episode_id] = old_val * self.discount + reward * (1 - self.discount)

        self.total_updates += 1

        # Decay epsilon (less exploration over time).
        rcfg = self.config.retrieval
        self.epsilon = max(rcfg.rl_epsilon_floor, self.epsilon * rcfg.rl_epsilon_decay)

    def batch_update(self, retrieved_ids: List[int], 
                     embeddings: List[np.ndarray], 
                     outcome_score: float):
        """Update all retrieved episodes based on the final outcome.
        
        Higher outcome_score → positive reward for retrieved episodes.
        Lower outcome_score → negative reward.
        """
        # Normalize reward: center around 0.5
        reward = (outcome_score - 0.5) * 2.0  # maps [0,1] to [-1,1]
        
        # Diminishing reward for later-retrieved episodes
        for rank, (eid, emb) in enumerate(zip(retrieved_ids, embeddings)):
            rank_discount = 1.0 / (1.0 + rank * 0.3)
            self.update(eid, emb, reward * rank_discount)

    def save(self):
        path = os.path.join(self.persist_dir, "rl_retriever.json")
        data = {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "episode_values": {str(k): v for k, v in self.episode_values.items()},
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save RL retriever: {e}")

    def _load(self):
        path = os.path.join(self.persist_dir, "rl_retriever.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.weights = np.array(data["weights"], dtype=np.float32)
            self.bias = float(data["bias"])
            self.episode_values = {int(k): v for k, v in data.get("episode_values", {}).items()}
            self.epsilon = data.get("epsilon", 0.1)
            self.total_updates = data.get("total_updates", 0)
        except Exception as e:
            logger.warning(f"Failed to load RL retriever: {e}")
