---
title: Self-Distillation
aliases: [online distillation, deep mutual learning, be your own teacher, early exit]
tags: [concept, deep-learning, distillation, efficient-ai]
---

## Definition
A distillation framework where knowledge is transferred dynamically without relying on a pre-trained, static external teacher—either between peer networks training simultaneously (**online distillation / mutual learning**) or within different depths of a single model (**self-distillation / intermediate exit supervision**).

## Why it's true
Standard offline distillation requires a two-phase pipeline: first train an enormous teacher model to convergence, then train a student model against frozen teacher outputs. This doubles computational overhead and demands large storage.

Self-distillation eliminates this bottleneck through two main mechanisms:
1. **Deep Mutual Learning (Peer Distillation):** Two or more networks start from random initialization and train concurrently on the same mini-batches. Each network optimizes standard cross-entropy on ground-truth labels plus a bidirectional KL divergence loss matching its peer's probability distribution:
   $$\mathcal{L}_{\Theta_1} = \mathcal{L}_{\text{CE}}(y, p_1) + D_{\text{KL}}(p_2 \,\|\, p_1)$$
   Because the models find distinct local minima and exploration trajectories, they exchange complementary insights throughout optimization. Remarkably, both models end up outperforming models of the same capacity trained in isolation.
2. **"Be Your Own Teacher" (Single-Model Self-Distillation):** A single network is augmented with auxiliary classification heads attached to intermediate residual blocks. The deepest classification head (which has the largest receptive field and highest capacity) serves as a dynamic online teacher for the shallower auxiliary heads:
   $$\mathcal{L} = \sum_{k=1}^K \left[ \mathcal{L}_{\text{CE}}(y, p_k) + \alpha T^2 D_{\text{KL}}(p_K^{(T)} \,\|\, p_k^{(T)}) \right]$$
   This supervision forces early layers to learn richer semantic features without waiting for gradients to backpropagate from the end of the network.

## Minimal example
Consider an image classifier with 12 transformer blocks:
- **Auxiliary Heads:** Attach linear classification heads after block 4, block 8, and block 12.
- **Training:** During the forward pass, block 12 predicts logits $z_{12}$. Blocks 4 and 8 are trained against ground truth plus softened logits from $z_{12}$.
- **Inference with Early Exit:**
  - For a clear, unambiguous photo of a golden retriever, the confidence at block 4 exceeds an early-exit threshold $\tau = 0.98$.
  - The model halts execution at block 4, skipping the remaining 8 blocks and saving $67\%$ of runtime latency.
  - Hard, ambiguous images proceed through all 12 blocks for full computational deliberation.

## Prerequisites
- [[concepts/knowledge-distillation|Knowledge Distillation]], [[concepts/soft-targets-and-temperature.md|Soft Targets and Temperature]]

## Common confusions
- **Mutual learning requires a larger teacher:** In Deep Mutual Learning, the peer models can have the exact same architecture and capacity (e.g., two identical ResNet-18s). They still benefit each other because their stochastic initialization and mini-batch sample order induce diverse feature representations.
- **Intermediate classifiers slow down deployment:** Auxiliary classification heads can be completely removed if dynamic early exiting is not desired, leaving the base model with improved accuracy at zero inference-time parameter or latency cost.

## Appears in
- [[lessons/2026-11-01-knowledge-distillation-principles]]: introduced as the solution for online peer learning and dynamic compute allocation.
