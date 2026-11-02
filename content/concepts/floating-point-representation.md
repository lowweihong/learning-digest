---
title: Floating-Point Representation
aliases: [IEEE 754, floating-point, FP32, FP16, BF16]
tags: [concept, computer-systems, numeric-formats]
---

## Definition
How computers store decimal numbers using scientific notation in binary, splitting bits into a sign ($+$ or $-$), an exponent (the overall scale), and a fraction (the fine detail).

## Why it's true
Neural network numbers range from tiny gradients like $0.00002$ to large weights like $50.0$. Regular whole numbers would either round the tiny numbers to zero or overflow on the large ones. Floating-point numbers solve this by scaling by powers of 2 ($(-1)^s \times (1 + f) \times 2^{e - \text{bias}}$), keeping relative accuracy steady across huge ranges.

## Minimal example
Writing the number $2.5$ in Brain Floating Point 16 (BF16):
- In scientific notation, $2.5 = 1.25 \times 2^1$.
- **Sign bit:** `0` (positive).
- **Exponent:** $1 + 127 = 128 = 10000000_2$.
- **Fraction:** $0.25 = 0100000_2$ (the leading $1.$ is automatic).
Final 16-bit binary: `0 10000000 0100000`.

## Prerequisites
- None

## Common confusions
- Confusing dynamic range (how big or small numbers can get) with precision (how many decimal digits you have): AI models care much more about dynamic range than precision. That is why BF16 keeps the full 8-bit exponent of FP32 while cutting the fraction bits in half.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: provides background on numerical data types before introducing low-bit quantization.
