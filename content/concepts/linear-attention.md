---
title: Linear Attention
aliases: [kernel attention, EfficientViT linear attention, associative attention, ReLU linear attention]
tags: [concept, efficient-attention, vision, mathematics, transformers]
---

## Definition
An efficient attention formulation that replaces the non-linear softmax similarity function with decomposed, non-negative feature maps ($\phi(Q)\phi(K)^T$), allowing matrix multiplication to be rearranged via associativity from $(QK^T)V$ to $Q(K^T V)$ and reducing computational complexity from quadratic $O(N^2 D)$ to linear $O(N D^2)$.

## Why it's true
Standard self-attention computes pairwise token similarities using the softmax operator:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{D}}\right) V$$

Here, $Q \in \mathbb{R}^{N \times D}$, $K \in \mathbb{R}^{N \times D}$, and $V \in \mathbb{R}^{N \times D}$, where $N$ is the sequence length and $D$ is the hidden dimension. Because the non-linear exponential function is trapped inside row-wise normalization, the full $N \times N$ attention matrix **must** be materialized:
$$\text{Softmax Arithmetic Cost} = O(N^2 D)$$

For high-resolution vision tasks ($N = 10,000+$ patches), this quadratic cost dominates execution time and memory bandwidth.

**Linear Attention** replaces the non-linear exponential kernel with a decomposable positive feature mapping $\phi(\cdot)$, such as $\phi(x) = \text{ReLU}(x)$:
$$\text{Sim}(q_i, k_j) = \phi(q_i) \phi(k_j)^T$$

Because matrix multiplication is associative ($A(BC) = (AB)C$), we can alter the order of operations:
$$\text{Output} = \frac{\phi(Q) \left(\phi(K)^T V\right)}{\phi(Q) \sum_{j=1}^N \phi(k_j)^T}$$

1. **Pre-aggregating Key-Value Context:** First, multiply $\phi(K)^T \in \mathbb{R}^{D \times N}$ by $V \in \mathbb{R}^{N \times D}$. This produces a compact global context matrix of shape $D \times D$.
2. **Linear Query Projection:** Next, multiply $\phi(Q) \in \mathbb{R}^{N \times D}$ by the $D \times D$ context matrix, resulting in an $N \times D$ output.

Because the hidden dimension $D$ (typically 64 or 128) is fixed and dramatically smaller than sequence length $N$ (thousands of tokens), the arithmetic complexity becomes $O(N D^2)$, scaling strictly **linearly with sequence length**.

## Minimal example
Consider an image with $N = 4,096$ patch tokens and head dimension $D = 64$:
- **Standard Softmax Attention:**
  - $Q K^T$ produces a $4,096 \times 4,096$ matrix ($16,777,216$ elements).
  - FLOPs: $2 \times 4096^2 \times 64 \approx 2.15 \times 10^9$ operations.
- **Linear Attention (EfficientViT):**
  - $K^T V$ produces a $64 \times 64$ matrix ($4,096$ elements).
  - Multiplying $Q$ by this $64 \times 64$ matrix requires $2 \times 4096 \times 64^2 \approx 3.35 \times 10^7$ operations.
  - Compute is reduced by **$64\times$**, and the multi-megabyte attention matrix is completely eliminated from GPU SRAM.

## Prerequisites
- None

## Common confusions
- **Linear attention produces identical focus to softmax:** Standard softmax acts as an exponential gate, creating sharp, spiky attention distributions around specific pixels. ReLU linear attention produces broader, more diffused feature distributions. EfficientViT compensates for this by inserting lightweight local depthwise convolutions alongside linear attention, restoring sharp local spatial detail while preserving global linear speed.

## Appears in
- [[lessons/2026-11-06-vision-transformers-perception-and-autoregressive-generation]]: used as the primary engine in EfficientViT and EfficientViT-SAM, accelerating Segment Anything by up to $49\times$.
