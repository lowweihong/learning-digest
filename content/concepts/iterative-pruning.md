---
title: Iterative Pruning
aliases: [iterative prune-and-retrain, multi-pass pruning]
tags: [concept, model-compression, pruning, optimization]
---

## Definition
A multi-stage compression process that alternates between pruning small increments of parameters and fine-tuning the surviving weights over several cycles, progressively reaching deep sparsity without collapsing model accuracy.

## Why it's true
Pruning a dense model to extreme sparsity in a single step (one-shot pruning) inflicts catastrophic damage on the network's internal representations, creating an error barrier too wide for standard gradient descent to cross. 

By removing weights in gentle increments (e.g., $15\%$ to $20\%$ per round) and retraining for several epochs in between, the surviving parameters can gradually adjust their numerical values to compensate for the missing connections, keeping the model inside a low-loss basin.

## Minimal example
On the AlexNet vision architecture:
1. **One-shot attempt:** Cutting $80\%$ of weights in a single pass drops classification accuracy from $57.2\%$ to near $0\%$; simple fine-tuning only partially recovers performance.
2. **Iterative pruning:**
   - Round 1: Prune $30\%$ of smallest weights $\rightarrow$ fine-tune for 2 epochs (accuracy stays at $57.2\%$).
   - Round 2: Prune another $30\%$ of remaining weights $\rightarrow$ fine-tune (accuracy $57.3\%$).
   - Round 3: Prune down to $90\%$ total sparsity $\rightarrow$ fine-tune (accuracy fully preserved at $57.2\%$).

By pacing the pruning schedule, the network achieves a $9\times$ to $10\times$ parameter reduction without dropping top-1 accuracy.

## Prerequisites
- [[concepts/magnitude-based-pruning|Magnitude-Based Pruning]]

## Common confusions
- **Static vs. dynamic masks:** In Song Han's classic deep compression pipeline, zeroed weights remain frozen at zero during fine-tuning (their gradients are masked out), meaning only surviving non-zero weights adapt. Modern variants (like dynamic pruning or sparse training) allow previously zeroed connections to regenerate if their gradient magnitudes spike during retraining.

## Appears in
- [[lessons/2026-11-01-pruning-and-sparsity-fundamentals]]: introduced as the key mechanism that pushed compression ratios from $50\%$ to $90\%$ without accuracy loss.
