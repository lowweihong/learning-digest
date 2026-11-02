---
title: "N:M Sparsity"
aliases: [2:4 sparsity, semi-structured sparsity, pattern-based sparsity]
tags: [concept, model-compression, hardware, pruning]
---

## Definition
A semi-structured sparsity pattern where out of every contiguous block of $M$ elements, at least $N$ elements must be zero.

## Why it's true
It resolves the fundamental tension between fine-grained flexibility and hardware efficiency:
1. **Algorithmic flexibility:** Enforcing local sparsity (e.g., in groups of 4) lets the algorithm retain the two most informative weights in every local neighborhood, preserving model accuracy much better than coarse channel removal.
2. **Hardware acceleration:** Dedicated hardware (such as NVIDIA Ampere, Hopper, and Blackwell Tensor Cores) features specialized data paths that read 2:4 compressed matrices directly. The hardware stores only the $2$ non-zero values and a compact 2-bit index per element, cutting memory bandwidth in half and executing matrix multiplies at $2\times$ theoretical peak math throughput.

## Minimal example
Consider an 8-weight row segment divided into two 4-weight blocks:

$$W = \underbrace{[0.1,\; 1.8,\; 0.3,\; -2.5]}_{\text{Block 1}}\;\; \underbrace{[-3.1,\; 0.05,\; 0.9,\; 0.2]}_{\text{Block 2}}$$

Under **2:4 sparsity** ($N=2, M=4$), we keep the two largest absolute weights in each block and zero the rest:
- **Block 1:** Keep $1.8$ (index 1) and $-2.5$ (index 3). Values $[1.8, -2.5]$, indices $[01_2, 11_2]$.
- **Block 2:** Keep $-3.1$ (index 0) and $0.9$ (index 2). Values $[-3.1, 0.9]$, indices $[00_2, 10_2]$.

The compressed data requires only $50\%$ of the original storage plus a negligible 2-bit metadata overhead per non-zero value.

## Prerequisites
- [[concepts/fine-grained-pruning|Fine-Grained Pruning]]
- [[concepts/structured-pruning|Structured Pruning]]

## Common confusions
- **Confusing 2:4 sparsity with arbitrary 50% sparsity:** If a weight tensor has $50\%$ random zeros scattered arbitrarily across its layout, standard Tensor Cores cannot accelerate it. Hardware 2:4 acceleration strictly requires the pattern to hold for **every single** 4-element vector slice along the inner matrix dimension.

## Appears in
- [[lessons/2026-11-01-pruning-and-sparsity-fundamentals]]: introduced as the practical industry standard for hardware-accelerated sparsity.
