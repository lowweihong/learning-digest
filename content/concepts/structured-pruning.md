---
title: Structured Pruning
aliases: [channel pruning, coarse-grained pruning, coarse-grained sparsity]
tags: [concept, model-compression, pruning, hardware]
---

## Definition
A pruning method that removes entire regular architectural blocks—such as complete convolution channels, output filters, attention heads, or matrix rows—from a neural network, leaving a smaller but strictly dense tensor.

## Why it's true
Standard computing hardware (NVIDIA GPUs, Intel/AMD CPUs, mobile NPUs) is optimized for dense General Matrix Multiply (GEMM) operations. When an entire input or output channel is deleted from a layer:
$$C_{\text{out}} \times C_{\text{in}} \times K \times K \longrightarrow C'_{\text{out}} \times C'_{\text{in}} \times K \times K$$

The remaining parameters form a regular, contiguous matrix. It requires zero sparse index overhead, causes no thread divergence, and executes immediately on standard, un-modified BLAS and CUDA kernels at reduced latency.

## Minimal example
In a convolutional layer producing $64$ feature maps, calculating $L_1$ norms reveals that $16$ filters have near-zero weights throughout their entire $3 \times 3$ receptive fields. 

Pruning those $16$ entire filters yields a new dense layer with only $48$ output filters. The subsequent layer's input channels also shrink from $64$ to $48$. The memory footprint and computation both drop by $25\%$ with immediate wall-clock speedup on any off-the-shelf GPU.

## Prerequisites
- [[concepts/magnitude-based-pruning|Magnitude-Based Pruning]]

## Common confusions
- **Assuming structured pruning matches fine-grained compression ratios:** Because structured pruning removes entire channels as an all-or-nothing package, it inevitably discards useful weights located inside a predominantly weak channel. Consequently, structured pruning typically degrades accuracy beyond $30\%$ to $50\%$ sparsity, whereas fine-grained pruning can often reach $80\%$ to $90\%$ sparsity on identical architectures.

## Appears in
- [[lessons/2026-11-01-pruning-and-sparsity-fundamentals]]: contrasted with fine-grained pruning as the hardware-friendly alternative.
