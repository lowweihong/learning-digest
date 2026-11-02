---
title: Static vs Dynamic Sparsity
aliases: [weight vs activation sparsity, static sparsity, dynamic sparsity]
tags: [concept, model-compression, hardware, pruning]
---

## Definition
The distinction between zero patterns that remain fixed across all inputs at compile time (**static sparsity**, e.g., pruned weights) versus zero patterns that fluctuate dynamically for every individual input sample at runtime (**dynamic sparsity**, e.g., ReLU activations or sparse attention).

## Why it's true
Understanding where zeros originate dictates hardware and compiler design:
1. **Static weight sparsity:** Weights are zeroed during training and offline compression. Because their memory coordinates never change regardless of user queries, compilers can pre-generate optimized compressed storage layouts (e.g., CSR, CSC, or 2:4 metadata) and fixed execution schedules ahead of time.
2. **Dynamic activation sparsity:** Activations are zeroed on the fly (e.g., negative inputs blocked by ReLU, uninformative tokens in sparse attention, or empty coordinates in 3D point clouds). Because different images or prompts light up completely different neurons, hardware cannot rely on pre-compiled layouts. It must inspect activations dynamically at runtime using hardware queues, leading-zero detectors, and adaptive grouping logic.

## Minimal example
In a fully connected layer with ReLU activations:
- **Weight Sparsity (Static):** $90\%$ of the weights are zeroed offline $\rightarrow$ eliminates $90\%$ of weight DRAM fetches, reducing computation by $10\times$.
- **Activation Sparsity (Dynamic):** $70\%$ of activations coming out of ReLU are zero for an incoming image $\rightarrow$ eliminates another $70\%$ of arithmetic operations.

Because both forms of sparsity are orthogonal, their arithmetic savings are **multiplicative**:
$$\text{Total Compute Reduction} = 10\times \times \frac{1}{1 - 0.7} = 10\times \times 3.3\times \approx 33\times$$
A specialized accelerator (like the Efficient Inference Engine, EIE) that skips both zero weights and zero activations executes $33\times$ fewer arithmetic operations on that layer.

## Prerequisites
- [[concepts/fine-grained-pruning|Fine-Grained Pruning]]

## Common confusions
- **Assuming dynamic sparsity saves DRAM bandwidth:** While static weight pruning directly shrinks model file size and off-chip weight memory transfers, dynamic activation sparsity primarily saves on-chip arithmetic cycles and register operations. Unless an accelerator specifically compresses intermediate activation tensors before evicting them to DRAM, dynamic sparsity does not automatically reduce off-chip memory footprint.

## Appears in
- [[lessons/2026-11-01-automated-pruning-and-hardware-sparsity]]: introduced as the fundamental divide governing software and hardware acceleration strategies for sparse models.
