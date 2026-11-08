---
title: State Space Models
aliases: [SSM, structured state space sequence models, Mamba, selective state spaces]
tags: [concept, architecture, sequence-modeling, mamba, linear-attention]
---

## Definition
A sequence modeling architecture inspired by classical linear dynamical systems that transforms input signals through a continuous hidden state discretized into recurrent updates ($h_t = \bar{A} h_{t-1} + \bar{B} x_t$), processing $N$ tokens in linear time $O(N)$ with constant $O(1)$ recurrent memory during generation while training in parallel on GPUs via associative prefix scans.

## Why it's true
Modern Transformers dominate NLP because self-attention allows any token to directly interact with any previous token. However, this creates two steep bottlenecks:
1. **Quadratic Training Cost:** Attention compute scales as $O(N^2)$ with sequence length $N$.
2. **Growing KV Cache:** Generation requires saving all past key-value states in memory, scaling as $O(N)$ memory per stream.

Recurrent Neural Networks (RNNs) solve both issues by compressing all past history into a fixed-size vector $h_t$, scaling with $O(N)$ compute and $O(1)$ memory. However, traditional RNNs suffer from two major flaws: they struggle with catastrophic forgetting over long sequences, and their sequential nature prevents parallel training on GPUs.

**State Space Models (SSMs)**, culminating in **Mamba (Selective State Spaces)**, bridge this gap:
1. **Continuous Dynamical Formulation:** The system models a continuous latent trajectory $h(t) \in \mathbb{R}^N$ governed by differential equations:
   $$\dot{h}(t) = A h(t) + B x(t), \quad y(t) = C h(t)$$
   Discretizing via a learned step size $\Delta$ yields discrete recurrent parameters:
   $$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$
2. **Selectivity (Content-Dependent Gating):** Classical SSMs used static $A, B, C$ matrices, restricting them to linear time-invariant (LTI) systems equivalent to convolutions that cannot selectively filter information based on context. Mamba makes $B$, $C$, and the step size $\Delta$ **dynamic functions of the current input $x_t$**. If a token is irrelevant, $\Delta \to 0$, causing $\bar{A} \to I$ and $\bar{B} \to 0$, effectively preserving the existing state without corrupting it with noise.
3. **Hardware-Efficient Parallel Scan:** Because linear recurrences are mathematically associative:
   $$(A_2, B_2 x_2) \circ (A_1, B_1 x_1) = (A_2 A_1, A_2 B_1 x_1 + B_2 x_2)$$
   the full sequence can be computed in parallel during training across GPU threads using a parallel prefix scan (work-efficient tree reduction) in $O(\log N)$ time, avoiding step-by-step sequential bottlenecks.

## Minimal example
Consider processing a 200,000-token legal document:
- **Transformer:** The model creates a $200,000 \times 200,000$ attention matrix ($4 \times 10^{10}$ dot products) and retains tens of gigabytes of KV cache in GPU memory. Generating token 200,001 is hundreds of times slower than generating token 2.
- **Mamba SSM:** The model maintains a compact recurrent state vector $h_t \in \mathbb{R}^{d \times d_{\text{state}}}$ (taking under 1 MB of SRAM). Processing token 200,001 requires the exact same fixed amount of compute and memory as processing token 2:
  $$h_{200001} = \bar{A}_{200001} h_{200000} + \bar{B}_{200001} x_{200001}$$

## Prerequisites
- None

## Common confusions
- **Mamba cannot perform exact retrieval like Transformers:** While pure SSMs compress history into a finite-dimensional state and can struggle with extreme verbatim multi-needle retrieval, modern hybrid models (such as Jamba) interleave Mamba layers with periodic Transformer attention layers, capturing linear-time throughput for 90% of the network while retaining exact attention retrieval where needed.
- **SSMs train sequentially like RNNs:** Because Mamba's recurrence equations are linear and associative, training is fully parallelized across GPU cores using hardware-accelerated prefix scans directly in fast SRAM.

## Appears in
- [[lessons/2026-11-04-long-context-llms-context-extension-and-sparse-attention]]: analyzed as an alternative sequence modeling architecture that breaks the quadratic compute and linear memory constraints of standard self-attention.
