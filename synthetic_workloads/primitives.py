"""
Synthetic data primitives.

- SyntheticItem / SyntheticQuery dataclasses
- Topic-cluster construction in unit hypersphere
- Item / query sampling around topic centers

Pure numpy; no LLM, no FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SyntheticItem:
    id: str
    topic_id: int
    embedding: np.ndarray  # d-dim, unit-norm
    created_at: int        # timestep when generated


@dataclass
class SyntheticQuery:
    id: str
    embedding: np.ndarray            # d-dim, unit-norm
    relevant_item_ids: List[str]     # ground truth
    timestep: int
    topic_id: int                    # for diagnostics


# ---------------------------------------------------------------------------
# Topic-space construction
# ---------------------------------------------------------------------------

def generate_topic_centers(
    n_topics: int,
    dim: int = 128,
    min_sep: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 5000,
) -> np.ndarray:
    """
    Sample n_topics unit-norm vectors on the d-sphere s.t. cosine
    similarity between any pair < (1 - min_sep).

    Falls back to "best-effort" if rejection runs out (rare for d=128).
    """
    if rng is None:
        rng = np.random.default_rng()

    centers: List[np.ndarray] = []
    tries = 0
    while len(centers) < n_topics and tries < max_tries:
        c = rng.standard_normal(dim)
        c = c / (np.linalg.norm(c) + 1e-12)
        ok = True
        for existing in centers:
            if float(np.dot(c, existing)) >= (1.0 - min_sep):
                ok = False
                break
        if ok:
            centers.append(c)
        tries += 1

    # Fallback: relax min_sep gradually if needed
    while len(centers) < n_topics:
        c = rng.standard_normal(dim)
        c = c / (np.linalg.norm(c) + 1e-12)
        centers.append(c)

    return np.stack(centers, axis=0)


def sample_item_embedding(
    topic_center: np.ndarray,
    intra_topic_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """topic_center + Gaussian noise, then unit-normalize."""
    noise = rng.standard_normal(topic_center.shape[0]) * intra_topic_std
    emb = topic_center + noise
    return emb / (np.linalg.norm(emb) + 1e-12)


def sample_query_embedding(
    topic_center: np.ndarray,
    query_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Query = topic center + small Gaussian noise (unit-normalized)."""
    noise = rng.standard_normal(topic_center.shape[0]) * query_noise
    q = topic_center + noise
    return q / (np.linalg.norm(q) + 1e-12)
