---
title: NetAdapt
aliases: [NetAdapt algorithm, progressive hardware-aware pruning]
tags: [concept, model-compression, hardware, pruning]
---

## Definition
An iterative, empirical pruning algorithm that progressively removes channels layer-by-layer to satisfy a global hardware metric (such as measured latency or battery energy) by taking incremental resource-reduction steps guided by a pre-built device lookup table.

## Why it's true
Pruning models directly to a target latency on real mobile devices is hampered by two issues:
1. Connecting a physical phone or microcontroller to evaluate thousands of candidate network configurations in real time is prohibitively slow.
2. Abstract metrics like FLOPs or parameter counts correlate poorly with true execution latency on modern cache-hierarchical hardware.

NetAdapt bridges this gap:
- It constructs an offline **latency lookup table** by profiling individual layer shapes directly on the target silicon once.
- In each optimization round, it targets a small latency drop $\Delta R$ (e.g., $10\text{ ms}$).
- For every layer, it uses the lookup table to determine how many channels must be pruned to yield that exact $\Delta R$ reduction.
- It performs short-term fine-tuning (e.g., $10\text{k}$ gradient steps) on each layer's candidate to approximate accuracy recovery.
- It permanently commits the candidate layer that achieves the highest post-tuning accuracy, repeating until the global latency target is achieved.

## Minimal example
Compressing MobileNet from $11\text{ ms}$ latency to $3\text{ ms}$ on a mobile GPU:
1. Target step: reduce latency by $\Delta R = 1\text{ ms}$.
2. Test pruning Layer 1 vs. Layer 2 vs. Layer $N$ such that each cut yields an exact $1\text{ ms}$ speedup according to the lookup table.
3. Fast fine-tune each candidate for $10\text{k}$ iterations:
   - Pruning Layer 1 $\rightarrow 68.2\%$ accuracy.
   - Pruning Layer 4 $\rightarrow 71.4\%$ accuracy (winner!).
4. Keep Layer 4 pruned, advance the network baseline to $10\text{ ms}$, and repeat until reaching $3\text{ ms}$.
5. Run a single long-term fine-tuning cycle at the end to fully restore accuracy.

## Prerequisites
- [[concepts/layer-sensitivity-analysis|Layer Sensitivity Analysis]]
- [[concepts/structured-pruning|Structured Pruning]]

## Common confusions
- **Full retraining vs. short-term fine-tuning:** NetAdapt cannot afford full convergence retraining within its inner search loop because it tests dozens of candidate layers per round. Instead, it relies on short-term retraining ($10\text{k}$ steps) as an accurate *proxy* for final post-recovery performance, saving hours of compute.

## Appears in
- [[lessons/2026-11-01-automated-pruning-and-hardware-sparsity]]: introduced as Vivian Sze's progressive algorithm for hardware-constrained channel pruning.
