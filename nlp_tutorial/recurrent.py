"""Recurrent neural network utilities for the tutorial package."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


class LSTM:
    """A minimal Long Short-Term Memory network.

    The implementation mirrors the mathematical formulation introduced by
    Hochreiter and Schmidhuber (1997). It is intentionally compact so that it
    can be read alongside the accompanying tutorial text.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed=13)
        # Gate weights: input, forget, cell, output
        self.weights_input = rng.normal(scale=0.1, size=(input_dim, 4 * hidden_dim))
        self.weights_hidden = rng.normal(scale=0.1, size=(hidden_dim, 4 * hidden_dim))
        self.bias = np.zeros(4 * hidden_dim)

    def _split_gates(self, gate_vector: np.ndarray) -> Tuple[np.ndarray, ...]:
        hidden = self.hidden_dim
        i = gate_vector[..., :hidden]
        f = gate_vector[..., hidden : 2 * hidden]
        g = gate_vector[..., 2 * hidden : 3 * hidden]
        o = gate_vector[..., 3 * hidden :]
        return i, f, g, o

    def _step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gates = x_t @ self.weights_input + h_prev @ self.weights_hidden + self.bias
        i, f, g, o = self._split_gates(gates)
        i = 1.0 / (1.0 + np.exp(-i))
        f = 1.0 / (1.0 + np.exp(-f))
        g = np.tanh(g)
        o = 1.0 / (1.0 + np.exp(-o))
        c_t = f * c_prev + i * g
        h_t = o * np.tanh(c_t)
        return h_t, c_t

    def forward(self, inputs: Sequence[np.ndarray], initial_state: Tuple[np.ndarray, np.ndarray] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the recurrent network on a sequence of input vectors."""

        if not inputs:
            raise ValueError("Inputs must be a non-empty sequence of vectors.")

        h_prev, c_prev = self._init_state(initial_state)
        outputs = []
        for x_t in inputs:
            if x_t.shape[-1] != self.input_dim:
                raise ValueError("Input dimensionality mismatch.")
            h_prev, c_prev = self._step(x_t, h_prev, c_prev)
            outputs.append(h_prev)
        return np.stack(outputs), h_prev, c_prev

    def _init_state(self, state: Tuple[np.ndarray, np.ndarray] | None) -> Tuple[np.ndarray, np.ndarray]:
        if state is not None:
            h_0, c_0 = state
            return h_0, c_0
        zeros = np.zeros(self.hidden_dim)
        return zeros.copy(), zeros.copy()

    def generate(self, seed: Iterable[np.ndarray], steps: int) -> np.ndarray:
        """Run the LSTM in free-running mode to generate hidden states."""

        _, h_prev, c_prev = self.forward(list(seed))
        outputs = []
        blank_input = np.zeros(self.input_dim)
        for _ in range(steps):
            h_prev, c_prev = self._step(blank_input, h_prev, c_prev)
            outputs.append(h_prev)
        return np.stack(outputs)


__all__ = ["LSTM"]
