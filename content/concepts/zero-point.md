---
title: Zero-Point
aliases: [quantization offset, zero offset, Z]
tags: [concept, quantization, machine-learning]
---

## Definition
The whole number (integer) that represents real zero ($0.0$) in a quantized network, ensuring zero-padding and turned-off neurons stay exactly zero with zero error.

## Why it's true
Neural networks rely heavily on exact zeros: convolutional layers pad image borders with zeros, and ReLU turns off negative activations to zero. If real zero mapped to a decimal approximation like $0.02$, every image border would glow with artificial noise. The zero-point $Z$ anchors continuous zero to an exact integer slot so zero stays zero.

## Minimal example
Suppose a layer's decimals range from $-1.08$ to $+2.12$, mapped to a 2-bit integer range $[-2, 1]$ with step size $S = 1.07$:
$$Z = \text{round}\left(Q_{\min} - \frac{R_{\min}}{S}\right) = \text{round}\left(-2 - \frac{-1.08}{1.07}\right) = \text{round}(-2 + 1.009) = -1$$
When the integer is $q = -1$, the real number is exact zero:
$$r = 1.07 \times (-1 - (-1)) = 0.0$$

## Prerequisites
- None

## Common confusions
- Thinking $Z$ can be a decimal: $Z$ must be an integer because hardware adds and subtracts it directly inside integer registers.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: used in the affine mapping formulation of linear quantization.
