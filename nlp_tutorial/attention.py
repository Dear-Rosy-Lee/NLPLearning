"""Attention mechanisms used throughout the tutorial package."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class Attention:
    """Scaled dot-product attention as described in Vaswani et al. (2017)."""

    def compute(
        self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray, mask: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if queries.ndim != 2 or keys.ndim != 2 or values.ndim != 2:
            raise ValueError("Queries, keys and values must be 2-D tensors of shape (seq_len, dim).")
        scale = float(np.sqrt(keys.shape[-1]))
        scores = (queries @ keys.T) / (scale + 1e-8)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError("Mask must have the same shape as the attention scores.")
            scores = np.where(mask, scores, -1e9)
        # Numerically stable softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-8
        attended = weights @ values
        return attended, weights


class SelfAttention(Attention):
    """Self-attention with learned query/key/value projections."""

    def __init__(self, model_dim: int, projection_dim: int | None = None, seed: int = 7) -> None:
        self.model_dim = model_dim
        self.projection_dim = projection_dim or model_dim
        rng = np.random.default_rng(seed)
        self.query_weights = rng.normal(scale=0.1, size=(model_dim, self.projection_dim))
        self.key_weights = rng.normal(scale=0.1, size=(model_dim, self.projection_dim))
        self.value_weights = rng.normal(scale=0.1, size=(model_dim, self.projection_dim))

    def _project(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = inputs @ self.query_weights
        keys = inputs @ self.key_weights
        values = inputs @ self.value_weights
        return queries, keys, values

    def __call__(self, inputs: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.model_dim:
            raise ValueError("Inputs must have shape (seq_len, model_dim).")
        queries, keys, values = self._project(inputs)
        return self.compute(queries, keys, values, mask)


class MaskedSelfAttention(SelfAttention):
    """Self-attention variant that prevents positions from attending to the future."""

    def __call__(self, inputs: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = inputs.shape[0]
        autoregressive_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        if mask is not None:
            if mask.shape != autoregressive_mask.shape:
                raise ValueError("Custom mask must match the autoregressive mask shape.")
            autoregressive_mask &= mask
        return super().__call__(inputs, mask=autoregressive_mask)


class MultiHeadSelfAttention:
    """Multi-head self-attention that aggregates several attention heads."""

    def __init__(self, model_dim: int, num_heads: int, seed: int = 21) -> None:
        if model_dim % num_heads != 0:
            raise ValueError("Model dimension must be divisible by the number of heads.")
        self.model_dim = model_dim
        self.num_heads = num_heads
        head_dim = model_dim // num_heads
        self.heads = [SelfAttention(model_dim, head_dim, seed + i) for i in range(num_heads)]
        rng = np.random.default_rng(seed)
        self.output_weights = rng.normal(scale=0.1, size=(model_dim, model_dim))
        self.output_bias = np.zeros(model_dim)

    def __call__(self, inputs: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        head_outputs = []
        attention_maps = []
        for head in self.heads:
            output, weights = head(inputs, mask=mask)
            head_outputs.append(output)
            attention_maps.append(weights)
        concatenated = np.concatenate(head_outputs, axis=-1)
        combined = concatenated @ self.output_weights + self.output_bias
        return combined, np.stack(attention_maps)


__all__ = ["Attention", "SelfAttention", "MaskedSelfAttention", "MultiHeadSelfAttention"]
