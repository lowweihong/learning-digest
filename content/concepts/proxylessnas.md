---
title: ProxylessNAS
aliases: [ProxylessNAS, direct hardware-aware search]
tags: [concept, automl, nas, hardware]
---

## Definition
A hardware-aware neural architecture search framework that searches directly on the target task (e.g., ImageNet) and target hardware without using proxy tasks (such as small proxy datasets, truncated training epochs, or proxy FLOP metrics), binarizing architecture paths to reduce GPU search memory from $O(N)$ to $O(1)$.

## Why it's true
Early NAS algorithms relied on proxy tasks because searching directly on full datasets was computationally prohibitive. However, models optimized for proxy tasks (like CIFAR-10) or proxy metrics (FLOPs) generalize poorly to real hardware: on GPUs, wide layers execute with flat latency due to massive SIMD parallelism, whereas on mobile CPUs, latency scales linearly with depth and channel count.

ProxylessNAS overcomes this by binarizing path selection during training:
- In each step, only one or two candidate operations are active in memory, dropping supernet memory footprint to that of a standard compact network.
- Architecture parameters $\alpha$ and network weights $w$ are updated alternately on the target dataset.
- Real measured latency from the target hardware is incorporated directly into the gradient loss function, producing specialized architectures that run up to $1.8\times$ faster than human-engineered baselines.

## Minimal example
Searching for a mobile vision model on ImageNet:
- **Proxy NAS baseline:** Search on CIFAR-10 for 20 epochs using FLOPs as the reward $\rightarrow$ transfers to ImageNet with high latency because low-FLOP depthwise convolutions suffer from kernel launch overhead on mobile GPUs.
- **ProxylessNAS:** Measures latency directly on a Qualcomm Snapdragon phone and searches on ImageNet directly. The algorithm discovers that the mobile GPU prefers shallower networks with large $7 \times 7$ kernels, whereas the CPU prefers deeper networks with smaller $3 \times 3$ kernels.

## Prerequisites
- [[concepts/differentiable-nas|Differentiable NAS]]

## Common confusions
- **FLOPs as a proxy for latency:** Minimizing FLOPs does not guarantee lower execution latency. A model with fewer FLOPs can run significantly slower if its architecture causes memory fragmentation, low arithmetic intensity, or frequent GPU kernel launches.

## Appears in
- [[lessons/2026-11-01-hardware-aware-nas-and-once-for-all-networks]]: introduced as the breakthrough eliminating proxy tasks in hardware-aware NAS.
