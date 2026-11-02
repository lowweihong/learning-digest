---
title: Zero-Point
aliases: [quantization offset, zero offset, Z]
tags: [concept, quantization, machine-learning]
---

## Definition
The integer value in quantized space that corresponds exactly to real-number zero in continuous floating-point space without any quantization error.

## Why it's true
Neural networks frequently operate on zero values—such as zero-padding in convolutional layers and ReLU activation clamping. If continuous zero maps to a non-zero floating-point approximation after dequantization, zero-padding introduces systematic bias and spurious activations, degrading network accuracy.

## Minimal example
Given a continuous range $[-1.08, 2.12]$ mapped to a 2-bit signed integer range $[-2, 1]$ with scale factor $S = 1.07$:
$$Z = \text{round}\left(Q_{\min} - \frac{R_{\min}}{S}\right) = \text{round}\left(-2 - \frac{-1.08}{1.07}\right) = \text{round}(-2 + 1.009) = -1$$
Here, real zero ($r=0$) dequantizes exactly: $r = S(q - Z) = 1.07 \times (-1 - (-1)) = 0$.

## Prerequisites
- None

## Common confusions
- Assuming $Z$ can be a floating-point number: $Z$ must be an integer because it lives in the quantized domain and is subtracted directly from integer activations/weights in integer hardware registers.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: used in the affine mapping formulation of linear quantization.
