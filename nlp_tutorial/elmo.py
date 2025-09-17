"""Simplified ELMo encoder for educational purposes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .recurrent import LSTM


class ELMo:
    """Bidirectional language model inspired by Peters et al. (2018).

    This class composes word representations from character or token-level
    embeddings by running a forward and backward LSTM. The outputs from the
    surface embedding layer and the two contextual layers are interpolated using
    learned scalar weights.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.forward_lstm = LSTM(embedding_dim, hidden_dim)
        self.backward_lstm = LSTM(embedding_dim, hidden_dim)
        # Three scalars corresponding to the token layer, forward LSTM and backward LSTM.
        self.layer_weights = np.zeros(3)
        self.gamma = 1.0  # scaling parameter from the original ELMo paper

    def contextualise(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        """Compute contextualised representations for a sentence.

        Parameters
        ----------
        embeddings:
            Pre-computed token embeddings for a sentence. Each element must be a
            NumPy array with dimensionality ``embedding_dim``.
        """

        if not embeddings:
            raise ValueError("ELMo requires a non-empty sequence of embeddings.")

        stacked_embeddings = np.stack(embeddings)
        forward_outputs, _, _ = self.forward_lstm.forward(embeddings)
        backward_outputs, _, _ = self.backward_lstm.forward(list(reversed(embeddings)))
        backward_outputs = backward_outputs[::-1]
        layers = np.stack([stacked_embeddings, forward_outputs, backward_outputs], axis=0)

        # Softmax-normalise the scalar weights.
        norm_weights = np.exp(self.layer_weights - np.max(self.layer_weights))
        norm_weights /= norm_weights.sum()

        # Weighted sum across layers, then scale by gamma.
        contextualised = self.gamma * np.tensordot(norm_weights, layers, axes=(0, 0))
        return contextualised


__all__ = ["ELMo"]
