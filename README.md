# NLP Learning Journey

This repository hosts an educational Python package that walks through the
history of neural approaches to Natural Language Processing (NLP). The
`nlp_tutorial` package contains intentionally compact implementations that
showcase the core mechanics behind seminal ideas—from the emergence of
recurrent neural networks to modern multimodal architectures.

## Why this project exists

The goal is to offer a hands-on narrative: you can import the components in a
notebook, inspect the code, and experiment with them to understand *how* each
innovation builds on the previous one. Every module favours readability over
raw performance so that you can trace the flow of tensors without needing a
large framework.

## Historical milestones covered by the package

| Era | Breakthrough | Module | Key idea |
| --- | --- | --- | --- |
| 2013 | Word2Vec | `Word2VecRNN` | Distributional semantics via co-occurrence prediction |
| 1997 & 2014 | LSTM & sequence modelling | `LSTM` | Long-term memory gates stabilise RNN training |
| 2018 | ELMo | `ELMo` | Deep, bidirectional contextual word representations |
| 2015–2017 | Attention | `Attention`, `SelfAttention`, `MaskedSelfAttention`, `MultiHeadSelfAttention` | Learn to focus on relevant tokens |
| 2017 | Positional coding | `PositionalEncoding` | Inject order information into attention-based models |
| 2017 | Transformer | `Transformer` | Stack of self-attention layers plus feedforward networks |
| 2018 | BERT | `BERT` | Bidirectional transformer pre-training with pooled outputs |
| 2021 | CLIP | `CLIP` | Align text and image representations through contrastive learning |

The modules are designed to be composed: later architectures reuse building
blocks introduced earlier so you can see how the ecosystem evolved.

## Package overview

```
nlp_tutorial/
├── __init__.py
├── attention.py
├── bert.py
├── clip.py
├── elmo.py
├── positional_encoding.py
├── recurrent.py
├── transformer.py
└── word_embeddings.py
```

### Word embeddings: `Word2VecRNN`
Implements a minimal co-occurrence based encoder. Build a vocabulary, train with
a toy recurrent update rule, and explore cosine similarity between word
vectors.

### Recurrent modelling: `LSTM`
Captures the gating mechanism that solved the vanishing gradient issue for
recurrent neural networks. The `forward` method mirrors the original
mathematical formulation.

### Contextual embeddings: `ELMo`
Stacked forward and backward LSTMs produce contextualised representations. The
class interpolates between the surface embedding layer and both contextual
layers, following the recipe from Peters et al. (2018).

### Attention family
`attention.py` introduces the evolution of attention mechanisms:

* `Attention` implements the scaled dot-product operation.
* `SelfAttention` learns query, key and value projections for a sequence.
* `MaskedSelfAttention` prevents tokens from attending to future positions—used
  for autoregressive language models.
* `MultiHeadSelfAttention` aggregates several attention heads and projects their
  concatenated output.

### Positional information: `PositionalEncoding`
Replicates the sinusoidal scheme popularised by the original transformer paper
so you can add deterministic position signals to any sequence of embeddings.

### Transformer encoder
`transformer.py` contains a lightweight encoder stack complete with residual
connections, layer normalisation and feed-forward networks. The
`Transformer` class provides an `encode` method that returns the contextualised
states alongside every layer's attention map.

### Pre-training revolutions: `BERT`
A BERT-like encoder built on top of the transformer stack. It includes token
and segment embeddings, pooled outputs and a simple classification head to
illustrate how pre-training powers downstream tasks.

### Multimodal contrastive learning: `CLIP`
The `CLIP` module shows how text and vision encoders can be trained jointly via
contrastive objectives. A transformer-based text branch and a linear image
projection meet in a shared embedding space, enabling zero-shot similarity
scoring.

## Getting started

```python
import numpy as np
from nlp_tutorial import (
    BERT,
    CLIP,
    ELMo,
    LSTM,
    PositionalEncoding,
    Transformer,
    Word2VecRNN,
)

# Word2Vec style embeddings
corpus = "the quick brown fox jumps over the lazy dog".split()
encoder = Word2VecRNN(embedding_dim=8)
encoder.build_vocab(corpus)
encoder.train_epoch(corpus)
print(encoder.most_similar("fox"))

# LSTM sequence modelling
lstm = LSTM(input_dim=8, hidden_dim=16)
inputs = [np.random.randn(8) for _ in range(5)]
outputs, _, _ = lstm.forward(inputs)

# Transformer encoding
transformer = Transformer(model_dim=16, num_heads=2, feedforward_dim=32)
inputs = np.random.randn(5, 16)
encoded, attentions = transformer.encode(inputs)

# BERT-style pooled representation
bert = BERT(vocab_size=30, model_dim=16, num_heads=2, feedforward_dim=32)
tokens = [1, 2, 3, 4]
logits, history = bert.classify(tokens)

# CLIP similarity
clip = CLIP(vocab_size=30, text_dim=16, vision_dim=32, embed_dim=16)
score, attention = clip.similarity(tokens, np.random.randn(32))
print("CLIP similarity:", score)
```

## Extending the tutorial

Each module favours clarity, leaving ample room for experimentation:

* Replace the numpy operations with your deep learning framework of choice to
  build trainable prototypes.
* Visualise the attention weights returned by the transformer and CLIP models
  to better understand how the networks focus on different parts of the input.
* Implement training loops for tasks such as language modelling, text
  classification or multimodal retrieval.

## License

This tutorial package is provided for educational use. Feel free to adapt it to
your own workshops, blog posts or classroom exercises.
