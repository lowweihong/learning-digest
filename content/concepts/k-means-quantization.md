---
title: K-Means Quantization
aliases: [weight sharing, codebook quantization, vector quantization]
tags: [concept, quantization, model-compression]
---

## Definition
A non-linear model compression method that groups continuous weights into $k = 2^n$ clusters using 1D K-means, replacing each weight with an $n$-bit cluster index that references a shared codebook of floating-point centroids.

## Why it's true
Weights in neural network layers often display clustering behavior and high redundancy, meaning thousands of distinct weights can be substituted by a single shared centroid with negligible impact on overall loss. By storing an $n$-bit integer per weight and a tiny lookup table of $k$ 32-bit floats, memory storage scales by roughly $\frac{n}{32}$ for large matrices.

## Minimal example
A $4 \times 4$ weight matrix (16 weights) is clustered into 4 centroids ($k=4 \implies n=2$ bits):
- Unquantized: $16 \text{ weights} \times 32 \text{ bits} = 512 \text{ bits}$ (64 bytes).
- Quantized: $16 \text{ indices} \times 2 \text{ bits} = 32 \text{ bits}$ (4 bytes) + codebook of $4 \times 32 \text{ bits} = 128 \text{ bits}$ (16 bytes) = 20 bytes total ($3.2\times$ reduction).
During fine-tuning, gradients are pooled by cluster: the gradient for centroid $c_i$ is the sum of gradients of all weights assigned to $c_i$.

## Prerequisites
- [[concepts/floating-point-representation|Floating-Point Representation]]

## Common confusions
- Assuming K-means quantization accelerates arithmetic: K-means quantization compresses memory storage and memory traffic, but standard arithmetic still occurs in full precision floating-point after dequantizing via lookup table.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: introduced as the weight sharing compression strategy in Deep Compression.
