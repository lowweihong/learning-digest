---
title: Magnitude-Based Pruning
aliases: [weight magnitude pruning, norm-based pruning, magnitude pruning]
tags: [concept, model-compression, pruning, deep-learning]
---

## Definition
A pruning heuristic that removes individual weights or weight groups whose absolute values (or norms) fall below a selected threshold, operating under the assumption that smaller weights contribute less to the model's final output.

## Why it's true
In standard neural network operations, an output neuron computes a weighted sum $y = \sum_i w_i x_i + b$. If a weight $w_i$ has a magnitude very close to zero, its product with any bounded input $x_i$ will produce a negligible contribution to the pre-activation sum. Setting that weight to zero alters the output activation by only a tiny perturbation $\Delta y = -w_i x_i$, making it the safest candidate to eliminate.

## Minimal example
Consider a single neuron with three inputs:
$$y = 10 x_0 - 8 x_1 + 0.1 x_2$$

If we must remove one parameter to save memory, we discard $w_2 = 0.1$. Even for a maximum input of $x_2 = 1.0$, removing $w_2$ shifts the total output by at most $0.1$, while permanently saving $33\%$ of the layer's weight storage.

For vector or channel pruning, the same principle extends to $L_1$ or $L_2$ norms:
- **$L_1$ norm:** $\sum_j |w_j|$ (sum of absolute values across the channel)
- **$L_2$ norm:** $\sqrt{\sum_j w_j^2}$ (Euclidean length of the weight filter)

Channels with the smallest aggregate norms are zeroed out first.

## Prerequisites
- None

## Common confusions
- **Magnitude vs. Saliency:** A small weight does not strictly guarantee zero importance. If a tiny weight sits on a steep loss cliff (high second-order curvature) or processes inputs with unusually large variance, zeroing it can hurt the model more than zeroing a larger weight on a flat valley. Despite this theoretical blind spot, magnitude pruning remains the most robust, widely adopted baseline in production because of its minimal computational overhead.

## Appears in
- [[lessons/2026-11-01-pruning-and-sparsity-fundamentals]]: introduced as the fundamental heuristic for deciding which weights to remove.
