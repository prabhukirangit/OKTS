"""Layer 3 / phase 1 — dense vector scorer with a deterministic offline fallback.

The default embedding hashes token and character n-gram features into a
fixed-dim vector — a lightweight, fully deterministic stand-in for a learned
embedding, so the dense signal works offline with no network calls and no
model downloads (required for CI). A real embedding function (e.g. calling an
API or a local model) can be injected via ``embed_fn`` for production use;
the fallback is what ships by default and what tests run against.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

import numpy as np

from okts.index.bm25 import tokenize

EmbedFn = Callable[[str], np.ndarray]

DEFAULT_DIM = 256
_NGRAM_SIZES = (3, 4, 5)
_NGRAM_WEIGHT = 0.5


def _hash_bucket(feature: str, dim: int) -> tuple[int, float]:
    """Map a string feature to ``(bucket_index, sign)`` via SHA1.

    SHA1 (not Python's built-in ``hash()``) so the mapping is stable across
    processes and platforms — ``hash()`` is salted per-process by default and
    would break determinism/reproducibility.
    """
    digest = hashlib.sha1(feature.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return idx, sign


def hashing_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Deterministic offline embedding: hashed bag-of-words + char n-grams.

    Word features capture exact/synonym overlap (OKT bodies spell synonyms
    out in prose by convention). Character n-gram features give partial
    credit for morphological variants (``issue`` / ``issues`` / ``tag`` /
    ``tagged``) that the plain BM25 tokenizer treats as unrelated terms. The
    result is L2-normalized so a dot product between two embeddings equals
    their cosine similarity.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for tok in tokenize(text):
        idx, sign = _hash_bucket(f"w:{tok}", dim)
        vec[idx] += sign
        padded = f"#{tok}#"
        for n in _NGRAM_SIZES:
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                idx, sign = _hash_bucket(f"n:{padded[i:i + n]}", dim)
                vec[idx] += _NGRAM_WEIGHT * sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


class DenseIndex:
    """Cosine-similarity search over hashed (or injected) embeddings."""

    def __init__(self, dim: int = DEFAULT_DIM, embed_fn: Optional[EmbedFn] = None):
        self.dim = dim
        self._embed_fn: EmbedFn = embed_fn or (lambda text: hashing_embed(text, dim=dim))
        self._doc_ids: list[str] = []
        self._matrix: Optional[np.ndarray] = None

    def fit(self, documents: dict[str, str]) -> None:
        """Build the index over ``{doc_id: text}``. Replaces any prior index."""
        self._doc_ids = list(documents)
        if not self._doc_ids:
            self._matrix = np.zeros((0, self.dim))
            return
        rows = [self._embed_fn(documents[doc_id]) for doc_id in self._doc_ids]
        self._matrix = np.vstack(rows)

    def score(self, query: str) -> dict[str, float]:
        """Return ``{doc_id: cosine_similarity}`` for every indexed document."""
        if not self._doc_ids or self._matrix is None or self._matrix.shape[0] == 0:
            return {}
        qvec = self._embed_fn(query)
        sims = self._matrix @ qvec
        return {doc_id: float(sim) for doc_id, sim in zip(self._doc_ids, sims)}
