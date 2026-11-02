---
title: Neural Architecture Search
aliases: [NAS, automated architecture design, AutoML architecture search]
tags: [concept, deep-learning, automl, hardware]
---

## Definition
An automated optimization process that searches a defined architectural design space to discover neural network structures (such as layer types, connections, widths, and depths) that maximize accuracy and efficiency under target hardware constraints.

## Why it's true
Manually designing neural network architectures requires months of expert intuition, trial-and-error, and empirical tuning. Furthermore, human designers heavily rely on uniform rules of thumb (such as doubling channels when halving spatial resolution) that are sub-optimal for heterogeneous hardware platforms.

NAS automates this discovery by defining:
1. **Search Space:** The set of valid candidate graph topologies, primitive operators, channel widths, and resolutions.
2. **Search Strategy:** An exploration algorithm (such as Reinforcement Learning, Evolutionary Search, or Differentiable Gradient Descent) that proposes candidate models.
3. **Performance Evaluator:** A scoring metric that assesses candidates against target objectives (accuracy, latency, memory footprint, energy).

## Minimal example
Instead of manually designing a vision network for an edge processor:
- **Search Space:** Choice between $3 \times 3$, $5 \times 5$, or $7 \times 7$ inverted residual blocks with expansion factors $\in \{3, 6\}$ across 20 layers.
- **Search Strategy:** An evolutionary algorithm mutates layer kernels and crosses over winning parent architectures.
- **Objective:** Maximize ImageNet top-1 accuracy under a hard limit of $\le 50\text{ ms}$ latency on an edge NPU.
The search discovers an asymmetric network that outperforms manually engineered networks like MobileNet by $2\%\text{–}3\%$ accuracy at identical latency.

## Prerequisites
- [[concepts/algorithm-hardware-co-design|Algorithm-Hardware Co-Design]]

## Common confusions
- **NAS vs. Pruning:** Pruning starts with an existing, fully trained human architecture and removes redundant parameters. NAS designs the network topology and operator allocations from scratch before training begins.

## Appears in
- [[lessons/2026-11-01-neural-architecture-search-foundations]]: introduced as the overarching paradigm shifting from post-hoc compression to automated ground-up model design.
