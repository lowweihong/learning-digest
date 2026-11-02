---
title: Deep Compression Pipeline
aliases: [Deep Compression, three-stage compression, Han et al. pipeline]
tags: [concept, model-compression, deep-learning]
---

## Definition
A three-step compression pipeline—pruning, quantization, and Huffman coding—that shrinks deep learning models by 30× to 50× without losing accuracy.

## Why it's true
Each step removes a different kind of waste:
1. **Pruning** cuts out useless connections between neurons.
2. **Quantization** reduces the number of bits each remaining number needs.
3. **Huffman coding** assigns shorter binary codes to the most frequent numbers.
Pruning must happen before quantization because removing connections first leaves the remaining weights free to adjust and heal accuracy during fine-tuning.

## Minimal example
Compressing the classic AlexNet vision model:
1. **Step 1 (Pruning):** Delete weights close to zero and retrain the model $\rightarrow$ **9× smaller**.
2. **Step 2 (Quantization):** Group surviving weights into 4-bit and 2-bit buckets $\rightarrow$ **27× to 31× smaller**.
3. **Step 3 (Huffman Coding):** Use shorter binary codes for common bucket IDs $\rightarrow$ **35× to 49× smaller**.
When combined with compact architectures like SqueezeNet, the overall model shrinks by **510×**, reducing a 240 MB network to just 0.47 MB.

## Prerequisites
- [[concepts/k-means-quantization|K-Means Quantization]]

## Common confusions
- Trying to prune after quantizing: If you quantize first, you lock weights into a tiny set of shared buckets. Pruning afterwards breaks the buckets, leaving too few degrees of freedom to heal the model's accuracy.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: presented as the overarching framework unifying pruning, weight sharing, and entropy coding.
