---
title: Differentiable NAS
aliases: [DARTS, continuous relaxation NAS, gradient-based NAS]
tags: [concept, automl, optimization, deep-learning]
---

## Definition
A neural architecture search method that replaces discrete operator choices with a continuous relaxation over a directed acyclic supernet, optimizing both network weights and architectural selection probabilities simultaneously via standard gradient descent.

## Why it's true
Early reinforcement learning and evolutionary NAS algorithms required sampling and training thousands of candidate networks to convergence from scratch, consuming tens of thousands of GPU hours (e.g., Zoph & Le's original NAS required 800 GPUs running for 28 days).

Differentiable NAS (e.g., DARTS, FBNet, ProxylessNAS) overcomes this by constructing a single over-parameterized "supernet" where every edge between layers contains parallel candidate operations ($3 \times 3$ conv, $5 \times 5$ conv, identity, pooling). The output of edge $(i, j)$ is a softmax-weighted sum:
$$\bar{o}^{(i, j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i, j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i, j)})} o(x)$$

Because the architecture parameters $\alpha$ are continuous scalars, they can be optimized using standard backpropagation alongside the network weights $w$.

## Minimal example
Making hardware latency differentiable in ProxylessNAS / FBNet:
1. Pre-measure the physical hardware latency of each primitive operator on the target chip: $\text{Lat}(3 \times 3) = 2.1\text{ ms}$, $\text{Lat}(5 \times 5) = 4.8\text{ ms}$, $\text{Lat}(\text{Identity}) = 0.1\text{ ms}$.
2. Define the expected layer latency as the probability-weighted sum:
   $$\mathbb{E}[\text{Latency}] = \sum_o p_o \cdot \text{Lat}(o)$$
3. Define the total loss function:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(w, \alpha) + \lambda \cdot \sum_{\text{layers}} \mathbb{E}[\text{Latency}]$$
If an operator is slow ($5 \times 5$), increasing its probability $p_o$ heavily penalizes the loss. Gradient descent naturally down-weights slow operators unless they provide an indispensable boost to task accuracy, driving search time from weeks down to a single day on a single GPU.

## Prerequisites
- [[concepts/neural-architecture-search|Neural Architecture Search]]

## Common confusions
- **Memory footprint during search:** A naive DARTS supernet holds all candidate operations and their intermediate activations in GPU memory simultaneously, creating severe memory bottlenecks. Advanced variants (like ProxylessNAS) binarize the path selection during training, evaluating only one or two sampled paths per step to fit within standard GPU memory.

## Appears in
- [[lessons/2026-11-01-neural-architecture-search-foundations]]: introduced as the formulation that made latency differentiable and democratized NAS compute.
