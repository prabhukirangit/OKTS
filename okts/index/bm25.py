"""Layer 3 / phase 1 — pure-Python BM25 (Okapi) over ``OKTConcept.match_text()``.

No dependencies beyond the stdlib. This is the ranking signal the baseline
retriever (``FlatBM25Retriever``) uses on its own, and one half of the hybrid
signal ``GraphAwareRetriever`` fuses with the dense scorer.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + split on runs of non-alphanumeric characters.

    Deliberately simple: no stemming, no stopword removal. The OKT body is
    retrieval text that already spells out synonyms in prose (see CLAUDE.md
    invariant #3), so a plain tokenizer is enough to pick them up.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 ranking over a fixed corpus of ``{doc_id: text}`` documents.

    Standard formulation (Robertson/Sparck Jones), with the ``+1`` idf
    smoothing that keeps idf non-negative for very common terms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_freqs: dict[str, Counter] = {}
        self._doc_lens: dict[str, int] = {}
        self._df: Counter = Counter()
        self._n_docs: int = 0
        self._avgdl: float = 0.0

    def fit(self, documents: dict[str, str]) -> None:
        """Build the index over ``{doc_id: text}``. Replaces any prior index."""
        self._doc_ids = list(documents)
        self._doc_freqs = {}
        self._doc_lens = {}
        self._df = Counter()
        for doc_id, text in documents.items():
            counts = Counter(tokenize(text))
            self._doc_freqs[doc_id] = counts
            self._doc_lens[doc_id] = sum(counts.values())
            for term in counts:
                self._df[term] += 1
        self._n_docs = len(self._doc_ids)
        self._avgdl = (sum(self._doc_lens.values()) / self._n_docs) if self._n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> dict[str, float]:
        """Return ``{doc_id: bm25_score}`` for every indexed document.

        Documents sharing no query term with the corpus score ``0.0`` (never
        negative — the ``+1`` smoothed idf can't go below ``ln(1) == 0``).
        """
        if not self._doc_ids:
            return {}
        scores = {doc_id: 0.0 for doc_id in self._doc_ids}
        q_terms = tokenize(query)
        if not q_terms:
            return scores
        for term in set(q_terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_id in self._doc_ids:
                tf = self._doc_freqs[doc_id].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom
        return scores
