---
title: AutoML for Model Compression
aliases: [AMC, automated pruning, RL-based compression]
tags: [concept, model-compression, automl, reinforcement-learning]
---

## Definition
An automated optimization framework that uses a reinforcement learning agent to search for the optimal continuous per-layer compression and pruning ratios under a global resource constraint.

## Why it's true
Manually tuning per-layer pruning ratios through sensitivity sweeps requires weeks of trial-and-error by specialized engineers, and rules of thumb treat layers independently. By framing compression as a continuous Markov Decision Process:
- **State:** Layer configuration embedding (layer index, channel count, spatial kernel dimensions, FLOP count).
- **Action:** A continuous scalar pruning ratio $a_t \in [0, 1]$ chosen by a policy agent (such as Deep Deterministic Policy Gradient, DDPG).
- **Reward:** $R = -\text{Error} \times \log(\text{FLOPs})$ or validation accuracy under a hard latency lookup table constraint.

The agent automatically discovers cross-layer parameter interactions in hours rather than weeks, allocating heavy pruning to redundant layers and preserving sensitive layers.

## Minimal example
When applied to ResNet-50 on ImageNet:
- **Human engineer baseline:** Took roughly one week of manual sensitivity tuning to prune ResNet-50 down to $29\%$ of its original parameters without accuracy loss.
- **AMC agent:** In just four hours of automated policy search, AMC pruned ResNet-50 down to $20\%$ density while matching baseline accuracy.

When inspecting the agent's chosen policy, it automatically learned an architectural principle: $1 \times 1$ bottleneck convolutions (having only 1 parameter per channel) were pruned gently, while wide $3 \times 3$ convolutions were pruned aggressively.

## Prerequisites
- [[concepts/layer-sensitivity-analysis|Layer Sensitivity Analysis]]

## Common confusions
- **Rewarding FLOPs linearly vs. logarithmically:** Because accuracy scales logarithmically with compute (doubling FLOPs only yields diminishing returns in accuracy), defining reward linearly ($-\text{Error} \times \text{FLOPs}$) over-penalizes deep models. Scaling compute logarithmically ($-\text{Error} \times \log(\text{FLOPs})$) aligns the agent's reward landscape with real-world empirical scaling laws.

## Appears in
- [[lessons/2026-11-01-automated-pruning-and-hardware-sparsity]]: introduced as Song Han's reinforcement-learning framework that replaced manual human sensitivity tuning.
