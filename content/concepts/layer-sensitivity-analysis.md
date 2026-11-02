---
title: Layer Sensitivity Analysis
aliases: [pruning sensitivity analysis, per-layer sensitivity, sensitivity curve]
tags: [concept, model-compression, pruning, optimization]
---

## Definition
An empirical diagnostic method that measures how much overall model accuracy drops when pruning a single layer across a range of sparsity ratios while keeping all other layers unpruned.

## Why it's true
Neural network layers exhibit vastly different degrees of parameter redundancy. Early layers (extracting basic low-level features) and compact $1 \times 1$ pointwise convolutions contain little slack, meaning even slight pruning degrades network representations. In contrast, deep layers and wide $3 \times 3$ convolutional filters often possess massive redundancy. Plotting individual layer sensitivity curves reveals the exact non-uniform pruning ratio each layer can tolerate before accuracy falls off a cliff.

## Minimal example
In a 5-layer convolutional network:
1. Leave Layers 1, 2, 4, 5 untouched at $100\%$ density.
2. Sweep Layer 3 from $10\%$ to $90\%$ sparsity and record validation accuracy at each step:
   - At $50\%$ sparsity: $75.2\%$ accuracy (baseline is $75.4\%$).
   - At $80\%$ sparsity: $74.9\%$ accuracy (robust!).
3. Repeat for Layer 1:
   - At $30\%$ sparsity: $75.1\%$ accuracy.
   - At $50\%$ sparsity: $61.0\%$ accuracy (highly sensitive!).

By establishing a horizontal error tolerance threshold (e.g., allow $\le 0.5\%$ accuracy loss), the intersection with each layer's curve determines its custom pruning budget: Layer 3 can be pruned to $80\%$ sparsity, while Layer 1 is capped at $20\%$.

## Prerequisites
- [[concepts/magnitude-based-pruning|Magnitude-Based Pruning]]
- [[concepts/structured-pruning|Structured Pruning]]

## Common confusions
- **Assuming layer independence:** Sweeping one layer at a time assumes layers do not compound errors. In reality, pruning two moderately sensitive layers simultaneously often causes a catastrophic accuracy drop that neither curve predicted in isolation.

## Appears in
- [[lessons/2026-11-01-automated-pruning-and-hardware-sparsity]]: introduced as the fundamental manual baseline for discovering non-uniform layer sparsity ratios.
