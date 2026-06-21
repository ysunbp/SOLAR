"""
FIFO (First-In-First-Out) Memory Agent

Extends the Embedder-based memory system with a capacity limit.
When memory is full, the oldest entry is evicted (FIFO order).

This is the simplest eviction baseline:
- Always stores every turn (like greedy/embedder)
- When full, removes the oldest entry regardless of its value
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class FIFOAgentConfig(EmbedderAgentConfig):
    """Configuration for FIFO Agent."""
    capacity: int = Field(
        default=500,
        description="Maximum number of memory entries. Oldest evicted when full."
    )


class FIFOAgent(EmbedderAgent):
    """
    FIFO Memory Agent.

    Stores every dialog turn. When capacity is reached, evicts the oldest entry.
    Uses the same embedding/retrieval as EmbedderAgent.
    """

    def __init__(self, config: FIFOAgentConfig = FIFOAgentConfig()):
        super().__init__(config)
        self.fifo_config = config
        self.total_candidates: int = 0
        self.total_evictions: int = 0
        self._active_count: int = 0  # maintained incrementally

    def _get_active_count(self) -> int:
        """Return cached active count (O(1))."""
        return self._active_count

    def _evict_oldest(self):
        """Evict the oldest non-deleted entry (FIFO)."""
        for i, meta in enumerate(self.metadata):
            if not meta.get("deleted", False):
                self.metadata[i]["deleted"] = True
                self.total_evictions += 1
                self._active_count -= 1
                return
    
    def add_memory_fifo(self, content: str, doc_id=None) -> bool:
        """
        Add memory with FIFO eviction when capacity is reached.
        Always stores (returns True), evicts oldest if full.
        """
        self.total_candidates += 1

        # Evict oldest if at capacity
        if self._get_active_count() >= self.fifo_config.capacity:
            self._evict_oldest()

        # Store
        if doc_id is None:
            doc_id = f"fifo_doc_{len(self.metadata)}"
        vector = self._embed(content)
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([vector]))
        self._active_count += 1
        return True

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """Add conversation with FIFO eviction."""
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            self.add_memory_fifo(content, doc_id)

    def get_fifo_metrics(self) -> Dict:
        """Return FIFO-specific metrics."""
        return {
            "total_candidates": self.total_candidates,
            "total_evictions": self.total_evictions,
            "active_memories": self._get_active_count(),
            "capacity": self.fifo_config.capacity,
        }
