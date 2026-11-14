---
title: Shifted Window Attention
aliases: [S-W-MSA, Swin attention, shifted window self-attention, windowed self-attention]
tags: [concept, vision, transformers, efficient-attention, swin]
---

## Definition
A localized attention mechanism that restricts self-attention to non-overlapping 2D spatial windows of fixed size $M \times M$, alternating between regular window partitions and half-window shifted partitions in consecutive layers to enable global information flow while keeping computational complexity strictly linear with image resolution.

## Why it's true
In standard Vision Transformers (ViT), global self-attention computes dot products between every pair of image patches. For an image with height $H$ and width $W$ partitioned into patches of size $P \times P$, the sequence length is $N = \frac{HW}{P^2}$. Computing full self-attention scales quadratically:
$$\text{Full Attention Complexity} = O(N^2 \cdot d) = O\left(\frac{H^2 W^2}{P^4} \cdot d\right)$$

For dense prediction tasks like object detection or high-resolution semantic segmentation ($1024 \times 1024$ or $2048 \times 2048$), $N$ exceeds thousands of tokens, causing quadratic compute and activation memory to blow up hardware limits.

**Window-based Self-Attention (W-MSA)** bounds this cost:
1. **Local Windows:** Partition the 2D token grid into fixed, non-overlapping windows containing $M \times M$ tokens (typically $M = 7$):
   $$\text{Window Complexity} = O\left(\frac{HW}{M^2 P^2} \cdot (M^2)^2 \cdot d\right) = O(N \cdot M^2 \cdot d)$$
   Because $M$ is constant, computation scales linearly $O(N)$ with image resolution.
2. **The Isolation Dilemma:** If window boundaries remain static across all layers, tokens in adjacent windows can never communicate. An object straddling a window boundary is split into disconnected representations.
3. **Shifted Window Partitioning (S-W-MSA):** In alternating layers, the window grid is shifted diagonally by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ tokens. A single window in Layer $l+1$ now encompasses tokens from four neighboring windows of Layer $l$. Across deep stacks, the effective receptive field expands across the entire image without ever paying for quadratic global attention.

## Minimal example
Consider an $8 \times 8$ grid of tokens partitioned into $4 \times 4$ windows ($M = 4$):
- **Layer 1 (Standard Windows):** Token $(3, 3)$ sits at the bottom-right corner of Window 1 ($[0..3, 0..3]$) and can only attend to tokens within that square. Token $(4, 4)$ sits across the boundary in Window 4 ($[4..7, 4..7]$).
- **Layer 2 (Shifted Windows):** The partition grid shifts by $(2, 2)$. A new window spans $[2..5, 2..5]$. Both Token $(3, 3)$ and Token $(4, 4)$ now sit inside the same window, directly exchanging visual features.

## Prerequisites
- None

## Common confusions
- **Shifted window attention requires padding at image borders:** Naive shifting produces smaller, irregular boundary windows at the image borders. Swin Transformer solves this using **cyclic shifting with masked attention**, shifting edge tokens around to the opposite side and applying an attention mask so disconnected image borders do not attend to each other, maintaining constant-size $M \times M$ tensor shapes without extra compute.

## Appears in
- [[lessons/2026-11-06-vision-transformers-perception-and-autoregressive-generation]]: introduced as the core architectural building block of Swin Transformer for linear-complexity visual perception.
