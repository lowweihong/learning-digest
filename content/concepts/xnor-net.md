---
title: XNOR-Net
aliases: [binary neural network, 1-bit quantization, popcount convolution]
tags: [concept, model-compression, quantization, hardware]
---

## Definition
An extreme 1-bit neural network architecture where both weights and activations are binarized into $\{-1, +1\}$, transforming floating-point matrix multiplications into bitwise XNOR logic and bit-counting (popcount) instructions.

## Why it's true
In a standard linear layer, computing an output activation requires multiply-accumulate operations:
$$y = \sum_{i=1}^{N} w_i x_i$$

When weights and activations are constrained to binary states $w_i, x_i \in \{-1, +1\}$:
- $(+1) \times (+1) = +1$
- $(-1) \times (-1) = +1$
- $(+1) \times (-1) = -1$
- $(-1) \times (+1) = -1$

If we encode $+1$ as boolean `1` and $-1$ as boolean `0`, the multiplication truth table matches standard boolean **XNOR** exactly (output is `1` if inputs match, `0` if they differ).

The summation over $N$ binary products can be evaluated by:
1. Executing bitwise `XNOR` across 32-bit or 64-bit word registers.
2. Counting the number of set bits (`1`s) using a hardware **popcount** instruction.
3. Converting the count back to arithmetic:
   $$W \cdot X = 2 \cdot \text{popcount}(W \text{ XNOR } X) - N$$

## Minimal example
Multiply two 4-element vectors:
$$W = [+1, -1, +1, +1], \quad X = [+1, +1, -1, +1]$$
- **True dot product:** $(1)(1) + (-1)(1) + (1)(-1) + (1)(1) = 1 - 1 - 1 + 1 = 0$.
- **XNOR-Net representation:**
  - Encoded $W = [1, 0, 1, 1]_2$
  - Encoded $X = [1, 1, 0, 1]_2$
  - Bitwise XNOR: $W \text{ XNOR } X = [1, 0, 0, 1]_2$
  - Popcount: $\text{popcount}([1, 0, 0, 1]_2) = 2$ set bits.
  - Result: $2 \times 2 - 4 = 0$ (exact match!).

Multiplication and addition hardware units are completely eliminated, replaced by single-cycle bitwise instructions with $32\times$ memory reduction and over $50\times$ theoretical energy efficiency.

## Prerequisites
- [[concepts/linear-quantization|Linear Quantization]]

## Common confusions
- **Binary Weight Networks vs. XNOR-Net:** In Binary Weight Networks (BWN), only weights are binarized; activations remain full-precision floating-point numbers, requiring floating-point adders. XNOR-Net binarizes **both** weights and activations, completely eliminating floating-point ALUs.

## Appears in
- [[lessons/2026-11-01-advanced-quantization-and-extreme-compression]]: introduced as the ultimate 1-bit computational extreme of quantization.
