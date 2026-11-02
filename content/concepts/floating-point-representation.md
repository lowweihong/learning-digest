---
title: Floating-Point Representation
aliases: [IEEE 754, floating-point, FP32, FP16, BF16]
tags: [concept, computer-systems, numeric-formats]
---

## Definition
A binary format representing real numbers by partitioning bits into three components: a sign bit, an exponent field (determining dynamic range with an applied bias), and a fraction/mantissa field (determining precision).

## Why it's true
Fixed-width integer formats have fixed dynamic range and constant absolute resolution. By allocating bits to an exponent, floating-point numbers represent values across many orders of magnitude ($(-1)^s \times (1 + f) \times 2^{e - \text{bias}}$ for normal numbers, and $(-1)^s \times f \times 2^{1 - \text{bias}}$ for subnormals), keeping relative precision roughly constant across disparate scales.

## Minimal example
In Brain Floating Point 16 (BF16), 16 bits are split into 1 sign bit, 8 exponent bits (bias = 127), and 7 fraction bits. The number 2.5 is represented as:
- Sign: $0$ (positive)
- Exponent: $1 + 127 = 128 = 10000000_2$ (since $2.5 = 1.25 \times 2^1$)
- Fraction: $0.25 = 0100000_2$ (the leading 1 is implicit)
Result: `0 10000000 0100000`.

## Prerequisites
- None

## Common confusions
- Conflating dynamic range with precision: Exponent bits dictate dynamic range (the maximum and minimum magnitudes), whereas mantissa/fraction bits dictate precision (number of significant digits). BF16 matches FP32's dynamic range while using half the total bits by sacrificing fraction bits.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: provides background on numerical data types before introducing low-bit quantization.
