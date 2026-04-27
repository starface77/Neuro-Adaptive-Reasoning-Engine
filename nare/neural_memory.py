"""
Titans/MIRAS-inspired Neural Memory Module.

Implements a lightweight neural network for dynamic long-term memory
without requiring PyTorch/TensorFlow. Uses pure NumPy.

Key concepts from the theory:
- Memory as a learnable MLP that adapts in real-time
- "Surprise" metric: gradient of divergence drives memory updates
- Huber loss for robustness to outliers (MIRAS/YAAD approach)
- Retention Gate: regularization-based forgetting

Architecture:
    Input (3072-dim embedding) → Hidden (256) → Output (256)
    The output is a compressed memory representation.
"""

import json
import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def _huber_loss(predicted: np.ndarray, target: np.ndarray, delta: float = 1.0) -> float:
    diff = predicted - target
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))

def _huber_grad(predicted: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    diff = predicted - target
    return np.where(np.abs(diff) <= delta, diff, delta * np.sign(diff))


class NeuralMemory:
    """Titans/MIRAS-inspired neural long-term memory.
    
    Learns compressed representations of episodic experience via
    an online-updated MLP. Uses surprise-driven gating and
    Huber-loss regularization for robustness.
    """

    def __init__(self, input_dim: int = 3072, hidden_dim: int = 256,
                 learning_rate: float = 0.001, retention_decay: float = 0.999,
                 persist_dir: str = "memory_store"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.retention_decay = retention_decay
        self.persist_dir = persist_dir

        # MLP: input → hidden → output
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Retention Gate weights (scalar per hidden unit)
        self.retention_gate = np.ones(hidden_dim, dtype=np.float32)

        # Statistics
        self.total_updates = 0
        self.avg_surprise = 0.0

        self._load()

    def forward(self, embedding: np.ndarray) -> np.ndarray:
        """Forward pass: embedding → compressed representation."""
        x = np.array(embedding, dtype=np.float32).flatten()[:self.input_dim]
        h = _relu(x @ self.W1 + self.b1)
        # Apply retention gate (attention-like masking)
        h = h * self.retention_gate
        out = _relu(h @ self.W2 + self.b2)
        return out

    def compute_surprise(self, embedding: np.ndarray,
                         target: Optional[np.ndarray] = None) -> float:
        """Heuristic novelty score for an input embedding.

        IMPORTANT — terminology disclaimer:
          This is *not* "surprise" in the Free-Energy / Bayesian sense
          (which would be -log p(o)). When ``target`` is None, we return
          the standard deviation of the network's output as a cheap
          proxy for "how non-uniform is the model's response"; when a
          target is provided, we return Huber loss between forward(x)
          and target. Both are bounded heuristics, useful for *ranking*
          novelty among recent episodes, but should not be interpreted
          as a probabilistic surprise.

        Returns a non-negative scalar; higher = more novel / less well
        predicted by the current weights.
        """
        x = np.array(embedding, dtype=np.float32).flatten()[:self.input_dim]
        output = self.forward(x)

        if target is None:
            return float(np.std(output))

        return float(_huber_loss(output, target))

    def update(self, embedding: np.ndarray, target: np.ndarray,
               importance: float = 1.0):
        """Online update: adjust weights based on new experience.
        
        Uses Huber loss gradient for robustness to outliers.
        importance scales the learning rate (surprise-driven).
        """
        x = np.array(embedding, dtype=np.float32).flatten()[:self.input_dim]
        target = np.array(target, dtype=np.float32).flatten()[:self.hidden_dim]

        # Forward pass with intermediate caching
        z1 = x @ self.W1 + self.b1
        h = _relu(z1)
        h_gated = h * self.retention_gate
        z2 = h_gated @ self.W2 + self.b2
        output = _relu(z2)

        # Backward pass with Huber gradient
        grad_output = _huber_grad(output, target) * _relu_deriv(z2)
        
        effective_lr = self.lr * importance

        # Update W2, b2
        grad_W2 = np.outer(h_gated, grad_output)
        self.W2 -= effective_lr * grad_W2
        self.b2 -= effective_lr * grad_output

        # Backprop to hidden
        grad_h_gated = grad_output @ self.W2.T
        grad_h = grad_h_gated * self.retention_gate * _relu_deriv(z1)

        # Update W1, b1
        grad_W1 = np.outer(x, grad_h)
        self.W1 -= effective_lr * grad_W1
        self.b1 -= effective_lr * grad_h

        # Update retention gate
        gate_grad = grad_h_gated * h
        self.retention_gate -= effective_lr * 0.1 * gate_grad
        self.retention_gate = np.clip(self.retention_gate, 0.01, 1.0)

        # Retention regularization: slow decay toward uniform
        self.retention_gate *= self.retention_decay
        self.retention_gate += (1 - self.retention_decay) * 1.0

        # Weight norm clipping
        for W in [self.W1, self.W2]:
            norm = np.linalg.norm(W)
            if norm > 100.0:
                W *= 100.0 / norm

        # Update statistics
        self.total_updates += 1
        surprise = _huber_loss(output, target)
        self.avg_surprise = 0.95 * self.avg_surprise + 0.05 * surprise

    def consolidate(self, episodes: List[Dict[str, Any]]):
        """Batch consolidation: learn from a set of episodes.
        
        High-surprise episodes get stronger updates (attention-driven).
        Routine episodes get minimal updates (prevents overwriting).
        """
        if not episodes:
            return

        surprises = []
        for ep in episodes:
            emb = ep.get('embedding')
            if emb is None:
                continue
            s = self.compute_surprise(np.array(emb, dtype=np.float32))
            surprises.append((s, ep))

        if not surprises:
            return

        # Sort by surprise (most surprising first)
        surprises.sort(key=lambda x: x[0], reverse=True)

        for surprise_val, ep in surprises:
            emb = np.array(ep['embedding'], dtype=np.float32)
            # Target: compressed representation of the solution
            target = self.forward(emb)  # Self-supervised target
            
            # Scale importance by surprise
            max_surprise = surprises[0][0] if surprises[0][0] > 0 else 1.0
            importance = surprise_val / max_surprise
            
            self.update(emb, target, importance=importance)

        logger.info(f"[NeuralMem] Consolidated {len(surprises)} episodes. "
                     f"Avg surprise: {self.avg_surprise:.4f}")

    def get_memory_representation(self, embedding: np.ndarray) -> np.ndarray:
        """Get the neural memory's compressed representation of an input."""
        return self.forward(embedding)

    def similarity_in_memory_space(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Compute cosine similarity in the learned memory space."""
        repr_a = self.forward(emb_a)
        repr_b = self.forward(emb_b)
        norm_a = np.linalg.norm(repr_a)
        norm_b = np.linalg.norm(repr_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(repr_a, repr_b) / (norm_a * norm_b))

    def save(self):
        path = os.path.join(self.persist_dir, "neural_memory.npz")
        try:
            np.savez(path,
                     W1=self.W1, b1=self.b1,
                     W2=self.W2, b2=self.b2,
                     retention_gate=self.retention_gate,
                     total_updates=np.array([self.total_updates]),
                     avg_surprise=np.array([self.avg_surprise]))
        except Exception as e:
            logger.warning(f"Failed to save neural memory: {e}")

    def _load(self):
        path = os.path.join(self.persist_dir, "neural_memory.npz")
        if not os.path.exists(path):
            return
        try:
            data = np.load(path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.retention_gate = data['retention_gate']
            self.total_updates = int(data['total_updates'][0])
            self.avg_surprise = float(data['avg_surprise'][0])
            logger.info(f"[NeuralMem] Loaded: {self.total_updates} updates, "
                         f"avg_surprise={self.avg_surprise:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load neural memory: {e}")
