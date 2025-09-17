"""Utility classes that mimic early word embedding models."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np


class Word2VecRNN:
    """A tiny educational implementation of a Word2Vec-style encoder.

    The class exposes only a handful of features that are useful for a
    conceptual tutorial:

    * building a vocabulary from an iterable of tokens
    * learning embeddings with a vanilla recurrent update rule
    * generating vectors for words and simple sentences

    The goal is to illustrate the ideas behind Word2Vec and how they paved
    the way for richer recurrent models.
    """

    def __init__(self, embedding_dim: int = 16, context_window: int = 2) -> None:
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.word_to_index: dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.recurrent_weights = np.zeros((embedding_dim, embedding_dim))
        self.recurrent_bias = np.zeros(embedding_dim)
        self._rng = np.random.default_rng(seed=0)

    def build_vocab(self, corpus: Iterable[str]) -> None:
        """Create a vocabulary and initialise the embedding matrix.

        Parameters
        ----------
        corpus:
            Any iterable that yields tokens. Tokens are assumed to already be
            pre-processed (lowercase, filtered) for simplicity.
        """

        counts = Counter(corpus)
        vocab = sorted(counts)
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        self.embeddings = self._rng.normal(
            loc=0.0, scale=0.1, size=(len(vocab), self.embedding_dim)
        )

    def _word_to_vec(self, word: str) -> np.ndarray:
        if not self.word_to_index:
            raise ValueError("The vocabulary is empty. Call `build_vocab` first.")
        if word not in self.word_to_index:
            raise KeyError(f"Unknown word: {word!r}")
        assert self.embeddings is not None  # for type-checkers
        return self.embeddings[self.word_to_index[word]]

    def encode_sentence(self, sentence: Sequence[str]) -> np.ndarray:
        """Encode a sentence by averaging its word embeddings."""

        vectors = [self._word_to_vec(token) for token in sentence]
        if not vectors:
            return np.zeros(self.embedding_dim)
        return np.mean(vectors, axis=0)

    def _context_indices(self, index: int, length: int) -> List[int]:
        start = max(0, index - self.context_window)
        end = min(length, index + self.context_window + 1)
        return [i for i in range(start, end) if i != index]

    def train_epoch(self, corpus: Sequence[str], learning_rate: float = 0.05) -> None:
        """Perform a very small training epoch with a recurrent update rule.

        The update rule is intentionally simplified: for each token we compute a
        context vector, run it through a recurrent transformation and nudge the
        token embedding in the direction of its neighbours. The implementation
        only aims to provide a sense of how co-occurrence learning works.
        """

        if self.embeddings is None:
            raise ValueError("Call `build_vocab` before training.")

        for idx, token in enumerate(corpus):
            if token not in self.word_to_index:
                continue
            neighbours = self._context_indices(idx, len(corpus))
            if not neighbours:
                continue

            context_vectors = [self._word_to_vec(corpus[i]) for i in neighbours]
            context = np.mean(context_vectors, axis=0)
            recurrent_update = np.tanh(context @ self.recurrent_weights + self.recurrent_bias)
            gradient = context + recurrent_update

            emb_index = self.word_to_index[token]
            self.embeddings[emb_index] += learning_rate * gradient

    def most_similar(self, word: str, top_k: int = 5) -> List[tuple[str, float]]:
        """Retrieve the nearest neighbours of a word based on cosine similarity."""

        target = self._word_to_vec(word)
        similarities: List[tuple[str, float]] = []
        assert self.embeddings is not None
        norms = np.linalg.norm(self.embeddings, axis=1)
        target_norm = np.linalg.norm(target)
        for other, index in self.word_to_index.items():
            if other == word:
                continue
            score = float(target @ self.embeddings[index] / (target_norm * norms[index] + 1e-8))
            similarities.append((other, score))
        similarities.sort(key=lambda pair: pair[1], reverse=True)
        return similarities[:top_k]


__all__ = ["Word2VecRNN"]
