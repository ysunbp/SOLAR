"""
Common memory state representations shared across all three approaches.
Provides the base abstraction for agent memory as a mathematical object.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    content: np.ndarray  # embedding vector
    timestamp: int
    source: str  # 'observation', 'feedback', 'compiled'
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: int = 0
    
    @property
    def dim(self) -> int:
        return self.content.shape[0]


@dataclass
class MemoryState:
    """
    The memory state x_t ∈ X at time t.
    Represented as a matrix where each row is a memory entry embedding.
    """
    entries: List[MemoryEntry] = field(default_factory=list)
    capacity: int = 100  # max entries (context window budget)
    embedding_dim: int = 128
    timestamp: int = 0
    
    @property
    def matrix(self) -> np.ndarray:
        """Return memory as a matrix (N x d)."""
        if not self.entries:
            return np.zeros((0, self.embedding_dim))
        return np.stack([e.content for e in self.entries])
    
    @property
    def size(self) -> int:
        return len(self.entries)
    
    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity
    
    def add(self, entry: MemoryEntry) -> Optional[MemoryEntry]:
        """Add entry, return evicted entry if at capacity."""
        evicted = None
        if self.is_full:
            # Default: evict least recently accessed
            idx = np.argmin([e.last_accessed for e in self.entries])
            evicted = self.entries.pop(idx)
        self.entries.append(entry)
        return evicted
    
    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-k most similar entries. Returns (index, similarity) pairs."""
        if not self.entries:
            return []
        M = self.matrix
        # Cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
        similarities = M_norm @ query_norm
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            self.entries[idx].access_count += 1
            self.entries[idx].last_accessed = self.timestamp
            results.append((idx, float(similarities[idx])))
        return results
    
    def distance_to(self, other: 'MemoryState') -> float:
        """
        Compute distance between two memory states.
        Used as the switching cost measure.
        """
        if self.size == 0 and other.size == 0:
            return 0.0
        if self.size == 0 or other.size == 0:
            return float(max(self.size, other.size))
        
        # Frobenius norm of difference in memory matrices (padded to same size)
        M1 = self.matrix
        M2 = other.matrix
        
        # Align by optimal matching (Hungarian) or simpler: compare kernel matrices
        K1 = M1 @ M1.T if M1.shape[0] > 0 else np.zeros((1, 1))
        K2 = M2 @ M2.T if M2.shape[0] > 0 else np.zeros((1, 1))
        
        # Pad to same size
        n = max(K1.shape[0], K2.shape[0])
        K1_pad = np.zeros((n, n))
        K2_pad = np.zeros((n, n))
        K1_pad[:K1.shape[0], :K1.shape[1]] = K1
        K2_pad[:K2.shape[0], :K2.shape[1]] = K2
        
        return np.linalg.norm(K1_pad - K2_pad, 'fro')
    
    def copy(self) -> 'MemoryState':
        """Deep copy of memory state."""
        import copy
        return copy.deepcopy(self)


@dataclass
class Feedback:
    """User feedback signal at time t."""
    query: np.ndarray  # the query that triggered feedback
    response_quality: float  # [-1, 1] score
    feedback_type: str  # 'explicit', 'action', 'implicit'
    suggested_content: Optional[np.ndarray] = None  # what should have been remembered
    timestamp: int = 0


class MemoryUpdatePolicy(ABC):
    """Abstract base class for memory update policies."""
    
    @abstractmethod
    def should_update(self, state: MemoryState, feedback: Feedback) -> bool:
        """Decide whether to update memory given current state and feedback."""
        pass
    
    @abstractmethod
    def compute_update(self, state: MemoryState, feedback: Feedback) -> MemoryState:
        """Compute the new memory state."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return algorithm-specific metrics."""
        pass
