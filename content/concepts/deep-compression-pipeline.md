---
title: Deep Compression Pipeline
aliases: [Deep Compression, three-stage compression, Han et al. pipeline]
tags: [concept, model-compression, deep-learning]
---

## Definition
A three-stage neural network compression workflow consisting of pruning, trained quantization (weight sharing via K-means), and Huffman coding to reduce model storage by 30× to 50× without degrading predictive accuracy.

## Why it's true
Each compression technique targets an independent form of redundancy: pruning eliminates unneeded connections (structural sparsity), quantization reduces bit-width per retained parameter (precision redundancy), and Huffman coding exploits non-uniform symbol frequencies among cluster indices (entropy redundancy). Executing pruning prior to quantization ensures that fine-tuning has maximum parameter flexibility before restricting values to fixed cluster centroids.

## Minimal example
On AlexNet:
1. **Pruning:** Reduces parameter count by ~10× (sparse weights retrained).
2. **K-Means Quantization:** Compresses remaining convolution weights to 8-bit / 4-bit and fully connected layers to 4-bit / 2-bit, reaching ~27×–31× overall compression.
3. **Huffman Coding:** Assigns variable-length prefix codes to frequent index symbols, boosting overall compression to ~35×.

## Prerequisites
- [[concepts/k-means-quantization|K-Means Quantization]]

## Common confusions
- Quantizing before pruning: Pruning after quantization is far harder because removing centroids destroys the shared codebook structure and leaves severely degraded capacity to recover accuracy through retraining.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: presented as the overarching framework unifying pruning, weight sharing, and entropy coding.
