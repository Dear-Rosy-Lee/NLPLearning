"""CLIP-style dual encoder."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .transformer import Transformer


class CLIP:
    """Contrastive Language-Image Pre-training inspired dual encoder."""

    def __init__(
        self,
        vocab_size: int,
        text_dim: int,
        vision_dim: int,
        embed_dim: int,
        num_heads: int = 2,
        feedforward_dim: int = 128,
        num_layers: int = 2,
        max_length: int = 77,
        seed: int = 2021,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim
        self.text_embeddings = rng.normal(scale=0.02, size=(vocab_size, text_dim))
        self.text_transformer = Transformer(
            model_dim=text_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            max_length=max_length,
            seed=seed,
        )
        self.text_projection = rng.normal(scale=0.02, size=(text_dim, embed_dim))
        self.vision_projection = rng.normal(scale=0.02, size=(vision_dim, embed_dim))

    def _attention_mask(self, mask: Iterable[int], length: int) -> np.ndarray:
        mask = np.asarray(list(mask), dtype=int)
        if mask.shape[0] != length:
            raise ValueError("Mask must match the number of tokens.")
        return np.outer(mask, mask).astype(bool)

    def encode_text(
        self, token_ids: Iterable[int], attention_mask: Iterable[int] | None = None
    ) -> Tuple[np.ndarray, list[np.ndarray]]:
        token_ids = np.asarray(list(token_ids), dtype=int)
        if token_ids.size == 0:
            raise ValueError("Text encoder requires at least one token.")
        if (token_ids < 0).any() or (token_ids >= self.vocab_size).any():
            raise ValueError("Token ids must be within the vocabulary range.")
        embeddings = self.text_embeddings[token_ids]
        if attention_mask is None:
            mask_matrix = None
        else:
            mask_matrix = self._attention_mask(attention_mask, token_ids.size)
        encoded, attention_history = self.text_transformer.encode(embeddings, mask=mask_matrix)
        pooled = encoded.mean(axis=0)
        projected = pooled @ self.text_projection
        return projected, attention_history

    def encode_image(self, image_features: Iterable[float]) -> np.ndarray:
        features = np.asarray(list(image_features), dtype=float)
        if features.shape[0] != self.vision_dim:
            raise ValueError("Image feature dimensionality mismatch.")
        return features @ self.vision_projection

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def similarity(
        self,
        token_ids: Iterable[int],
        image_features: Iterable[float],
        attention_mask: Iterable[int] | None = None,
    ) -> Tuple[float, list[np.ndarray]]:
        text_features, history = self.encode_text(token_ids, attention_mask)
        image_features = self.encode_image(image_features)
        text_features = self._normalise(text_features)
        image_features = self._normalise(image_features)
        score = float(text_features @ image_features)
        return score, history


__all__ = ["CLIP"]
