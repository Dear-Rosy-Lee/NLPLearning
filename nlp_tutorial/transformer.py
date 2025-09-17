"""Simple transformer encoder stack."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .attention import MultiHeadSelfAttention
from .positional_encoding import PositionalEncoding


class TransformerLayer:
    """A single transformer encoder block."""

    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int, seed: int = 100) -> None:
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads, seed=seed)
        rng = np.random.default_rng(seed)
        self.feedforward_weights_1 = rng.normal(scale=0.1, size=(model_dim, feedforward_dim))
        self.feedforward_weights_2 = rng.normal(scale=0.1, size=(feedforward_dim, model_dim))
        self.feedforward_bias_1 = np.zeros(feedforward_dim)
        self.feedforward_bias_2 = np.zeros(model_dim)
        self.attn_gamma = np.ones(model_dim)
        self.attn_beta = np.zeros(model_dim)
        self.ff_gamma = np.ones(model_dim)
        self.ff_beta = np.zeros(model_dim)

    @staticmethod
    def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalised = (x - mean) / np.sqrt(variance + eps)
        return gamma * normalised + beta

    def __call__(self, inputs: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        attn_output, attn_scores = self.self_attention(inputs, mask=mask)
        attn_residual = inputs + attn_output
        attn_normalised = self._layer_norm(attn_residual, self.attn_gamma, self.attn_beta)

        hidden = attn_normalised @ self.feedforward_weights_1 + self.feedforward_bias_1
        hidden = np.maximum(0.0, hidden)
        ff_output = hidden @ self.feedforward_weights_2 + self.feedforward_bias_2
        ff_residual = attn_normalised + ff_output
        output = self._layer_norm(ff_residual, self.ff_gamma, self.ff_beta)
        return output, attn_scores


class Transformer:
    """Stacked transformer encoder with positional encodings."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int = 2,
        max_length: int = 5000,
        seed: int = 123,
    ) -> None:
        self.model_dim = model_dim
        self.positional_encoding = PositionalEncoding(model_dim, max_length=max_length)
        self.layers: List[TransformerLayer] = []
        for layer_idx in range(num_layers):
            layer_seed = seed + layer_idx
            self.layers.append(TransformerLayer(model_dim, num_heads, feedforward_dim, seed=layer_seed))

    def encode(self, inputs: np.ndarray, mask: np.ndarray | None = None, add_positional: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.model_dim:
            raise ValueError("Inputs must have shape (seq_len, model_dim).")
        states = inputs
        if add_positional:
            states = self.positional_encoding.add_to(states)
        attention_history: List[np.ndarray] = []
        for layer in self.layers:
            states, attn = layer(states, mask=mask)
            attention_history.append(attn)
        return states, attention_history


__all__ = ["Transformer", "TransformerLayer"]
