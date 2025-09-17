"""Educational BERT-style encoder."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .transformer import Transformer


class BERT:
    """A compact BERT-like architecture for demonstration and experimentation."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int = 4,
        max_length: int = 128,
        seed: int = 999,
    ) -> None:
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        rng = np.random.default_rng(seed)
        self.token_embeddings = rng.normal(scale=0.02, size=(vocab_size, model_dim))
        self.segment_embeddings = rng.normal(scale=0.02, size=(2, model_dim))
        self.transformer = Transformer(
            model_dim=model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            max_length=max_length,
            seed=seed,
        )
        self.classifier_weights = rng.normal(scale=0.02, size=(model_dim, 2))
        self.classifier_bias = np.zeros(2)

    def _prepare_inputs(
        self, token_ids: Iterable[int], segment_ids: Iterable[int] | None, attention_mask: Iterable[int] | None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        token_ids = np.asarray(list(token_ids), dtype=int)
        if (token_ids < 0).any() or (token_ids >= self.vocab_size).any():
            raise ValueError("Token ids must be within the vocabulary range.")
        if segment_ids is None:
            segment_ids = np.zeros_like(token_ids)
        segment_ids = np.asarray(list(segment_ids), dtype=int)
        if segment_ids.shape != token_ids.shape:
            raise ValueError("Segment ids must match the length of token ids.")
        if attention_mask is None:
            attention_mask = np.ones_like(token_ids)
        attention_mask = np.asarray(list(attention_mask), dtype=int)
        if attention_mask.shape != token_ids.shape:
            raise ValueError("Attention mask must match the length of token ids.")
        attention_matrix = np.outer(attention_mask, attention_mask).astype(bool)
        embeddings = self.token_embeddings[token_ids] + self.segment_embeddings[segment_ids]
        return embeddings, token_ids, segment_ids, attention_matrix

    def encode(
        self,
        token_ids: Iterable[int],
        segment_ids: Iterable[int] | None = None,
        attention_mask: Iterable[int] | None = None,
    ) -> Tuple[np.ndarray, list[np.ndarray]]:
        embeddings, _, _, attention_matrix = self._prepare_inputs(token_ids, segment_ids, attention_mask)
        encoded, attention_history = self.transformer.encode(embeddings, mask=attention_matrix)
        return encoded, attention_history

    def pooled_output(
        self,
        token_ids: Iterable[int],
        segment_ids: Iterable[int] | None = None,
        attention_mask: Iterable[int] | None = None,
    ) -> np.ndarray:
        encoded, _ = self.encode(token_ids, segment_ids, attention_mask)
        return encoded[0]

    def classify(
        self,
        token_ids: Iterable[int],
        segment_ids: Iterable[int] | None = None,
        attention_mask: Iterable[int] | None = None,
    ) -> Tuple[np.ndarray, list[np.ndarray]]:
        encoded, attention_history = self.encode(token_ids, segment_ids, attention_mask)
        pooled = encoded[0]
        logits = pooled @ self.classifier_weights + self.classifier_bias
        return logits, attention_history


__all__ = ["BERT"]
