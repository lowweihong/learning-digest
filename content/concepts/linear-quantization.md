---
title: Linear Quantization
aliases: [uniform quantization, affine quantization, integer quantization, INT8 quantization]
tags: [concept, quantization, hardware-acceleration]
---

## Definition
An affine mapping between continuous real-valued numbers $r$ and discrete $n$-bit integers $q$ defined by $r = S(q - Z)$, where $S$ is a positive real scale factor and $Z$ is an integer zero-point.

## Why it's true
Many edge processors and embedded microcontrollers lack Floating-Point Units (FPUs) or expend substantial power on floating-point arithmetic. Because matrix multiplication and convolution are linear operations, substituting $r = S(q - Z)$ allows inner products to be computed almost entirely using integer multiply-accumulate (MAC) instructions and bit shifts, with scaling factors precomputed or folded.

## Minimal example
To quantize real values $r \in [-1.08, 2.12]$ to 2-bit signed integers $q \in [-2, 1]$:
1. Scale factor: $S = \frac{R_{\max} - R_{\min}}{Q_{\max} - Q_{\min}} = \frac{2.12 - (-1.08)}{1 - (-2)} = \frac{3.20}{3} \approx 1.07$.
2. Zero-point: $Z = \text{round}\left(-2 - \frac{-1.08}{1.07}\right) = -1$.
3. Quantizing $r = 1.49$: $q = \text{round}\left(\frac{1.49}{1.07}\right) + (-1) = 1 + (-1) = 0$.
4. Reconstructed: $\hat{r} = 1.07 \times (0 - (-1)) = 1.07$.

## Prerequisites
- [[concepts/floating-point-representation|Floating-Point Representation]]
- [[concepts/zero-point|Zero-Point]]

## Common confusions
- Confusing symmetric and asymmetric quantization: In symmetric quantization, the zero-point $Z$ is clamped to 0, simplifying hardware math ($r = S \cdot q$) at the expense of clipping or underutilizing dynamic range when real values are distributed asymmetrically.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: provides the mathematical derivation for integer-only GEMM and convolution.
