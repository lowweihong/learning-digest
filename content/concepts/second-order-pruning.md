---
title: Second-Order Pruning
aliases: [Hessian-based pruning, Optimal Brain Damage, OBD]
tags: [concept, model-compression, optimization, pruning]
---

## Definition
A pruning criterion that estimates the change in loss resulting from removing weights by computing a second-order Taylor series expansion of the objective function, ranking parameter importance by their diagonal Hessian curvature ($H_{ii} w_i^2$).

## Why it's true
When a neural network finishes training and converges to a local minimum, the first-order gradient $\mathbf{g} = \nabla_{\mathbf{w}} L$ approaches zero ($\mathbf{g} \approx 0$). Expanding the change in loss $\Delta L$ when perturbing weights by $\Delta \mathbf{w} = -\mathbf{w}$ yields:

$$\Delta L \approx \mathbf{g}^T \Delta \mathbf{w} + \frac{1}{2} \Delta \mathbf{w}^T \mathbf{H} \Delta \mathbf{w} \approx \frac{1}{2} \sum_i H_{ii} w_i^2$$

where $\mathbf{H}$ is the Hessian matrix (second partial derivatives of the loss) and off-diagonal cross-terms are assumed negligible. This reveals that weight importance (saliency) is governed not just by raw magnitude $w_i^2$, but by the local curvature $H_{ii}$.

## Minimal example
Suppose a network has two parameters under consideration:
- Parameter A: $w_A = 0.02$, with steep curvature $H_{AA} = 500$ (a sharp loss canyon).
- Parameter B: $w_B = 0.20$, with flat curvature $H_{BB} = 1$ (a wide loss valley).

Comparing impact on loss:
- Pruning A: $\Delta L_A \approx \frac{1}{2} (500) (0.02)^2 = 0.10$
- Pruning B: $\Delta L_B \approx \frac{1}{2} (1) (0.20)^2 = 0.02$

**Magnitude pruning** would mistakenly remove Parameter A because $|w_A| = 0.02 < 0.20$, increasing the loss five times more than necessary. **Second-order pruning** correctly preserves Parameter A and deletes Parameter B.

## Prerequisites
- [[concepts/magnitude-based-pruning|Magnitude-Based Pruning]]

## Common confusions
- **Tractability in deep models:** For modern networks with tens of millions to billions of parameters, calculating and inverting the full Hessian is computationally intractable ($O(N^2)$ memory, $O(N^3)$ operations). Practical implementations either assume a strictly diagonal Hessian (like LeCun's 1989 *Optimal Brain Damage*), approximate it using empirical Fisher information matrices, or fall back to magnitude heuristics.

## Appears in
- [[lessons/2026-11-01-pruning-and-sparsity-fundamentals]]: introduced as the mathematically rigorous formulation of parameter saliency.
