"""Sinusoidal positional encoding utilities."""

from __future__ import annotations

import numpy as np


class PositionalEncoding:
    """Implements the sinusoidal positional encoding used by transformers."""

    def __init__(self, model_dim: int, max_length: int = 5000) -> None:
        self.model_dim = model_dim
        self.max_length = max_length
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, model_dim, 2) * (-np.log(10000.0) / model_dim))
        encodings = np.zeros((max_length, model_dim))
        encodings[:, 0::2] = np.sin(position * div_term)
        encodings[:, 1::2] = np.cos(position * div_term)
        self.encoding_table = encodings

    def get(self, length: int) -> np.ndarray:
        if length > self.max_length:
            raise ValueError("Requested length exceeds the pre-computed encoding table.")
        return self.encoding_table[:length]

    def add_to(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[-1] != self.model_dim:
            raise ValueError("Inputs must have shape (seq_len, model_dim).")
        seq_len = inputs.shape[0]
        return inputs + self.get(seq_len)


__all__ = ["PositionalEncoding"]
