---
title: K-Means Quantization
aliases: [weight sharing, codebook quantization, vector quantization]
tags: [concept, quantization, model-compression]
---

## Definition
A compression method that groups similar weights into a small number of shared values (centroids), saving short bucket IDs instead of full decimal numbers.

## Why it's true
In a trained neural network, thousands of weights are nearly identical (e.g. `2.09`, `2.12`, `1.98`). Instead of storing every weight as a separate 32-bit decimal, we round them all to a single shared value (like `2.0`). Because there are only a few shared values (for example, 16 values for 4-bit), each weight only needs a tiny 4-bit ID pointing to which bucket it belongs to.

## Minimal example
Imagine a $4 \times 4$ patch of 16 weights grouped into 4 shared buckets ($k = 4$, which needs 2-bit IDs):
- **Without grouping:** 16 weights $\times$ 32 bits = 512 bits (64 bytes).
- **With grouping:** 16 bucket IDs $\times$ 2 bits (4 bytes) + 4 shared decimal numbers (16 bytes) = 20 bytes total (over 3× smaller).
On a full layer with millions of weights, the table overhead disappears, giving an **8× size reduction** for 4-bit buckets.

## Prerequisites
- [[concepts/floating-point-representation|Floating-Point Representation]]

## Common confusions
- Thinking K-means speeds up math on simple hardware: K-means saves memory traffic, but computing a layer still requires looking up the bucket value and multiplying using full decimal (floating-point) hardware.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: introduced as the weight sharing compression strategy in Deep Compression.
