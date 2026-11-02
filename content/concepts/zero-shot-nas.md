---
title: Zero-Shot NAS
aliases: [Z-NAS, training-free NAS, zero-cost proxies]
tags: [concept, automl, nas, deep-learning]
---

## Definition
A class of architecture evaluation metrics that rank the trainability, expressiveness, and final convergence accuracy of randomly initialized candidate neural networks without training them for a single gradient step.

## Why it's true
Even with supernets and weight-sharing, training or fine-tuning neural networks remains a major energy and time bottleneck. Zero-shot NAS seeks mathematical proxies computable at initialization (using random weights and a few validation batches) that correlate strongly with final trained accuracy:
1. **Perturbation Sensitivity (Z-NAS):** Measures how output activations respond to tiny input perturbations ($\log \|\Delta y\| / \|\Delta x\|$) combined with batch normalization scaling variance. A high score indicates the network has rich feature separation and avoids vanishing gradients.
2. **Gradient Sign Concordance (GradSign):** Counts the percentage of samples that produce identical gradient signs across layers. A high concordance indicates that sample-wise local minima are tightly clustered, leading to smooth optimization and fast convergence.

## Minimal example
Given two candidate network topologies $A$ and $B$:
- Compute the Z-score at random initialization:
  - Network $A$: $Z_A = 14.2$ (high output sensitivity, high batch norm variance).
  - Network $B$: $Z_B = 2.1$ (output collapses to a constant; low variance).
- Without training either network on ImageNet, the search engine discards Network $B$ immediately. Evaluating $Z$ takes 50 milliseconds on a single GPU, enabling the search of millions of architectures in minutes.

## Prerequisites
- [[concepts/neural-architecture-search|Neural Architecture Search]]

## Common confusions
- **Absolute accuracy vs. Relative rank correlation:** Zero-shot proxies do not predict what numerical accuracy a model will achieve (e.g. $76.2\%$). They only need to provide high **rank correlation** (Kendall's $\tau$ or Spearman's $\rho$) so the search algorithm reliably selects better architectures over worse ones.

## Appears in
- [[lessons/2026-11-01-hardware-aware-nas-and-once-for-all-networks]]: introduced as the training-free frontier for instantaneous architecture ranking.
