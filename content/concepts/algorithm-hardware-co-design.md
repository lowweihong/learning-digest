---
title: Algorithm-Hardware Co-Design
aliases: [co-design, hardware-aware design, system-algorithm co-design]
tags: [concept, computer-systems, efficiency]
---

## Definition
The practice of designing AI algorithms and hardware architectures together at the same time, tailoring the math to what silicon executes efficiently rather than designing them in isolation.

## Why it's true
Historically, software engineers designed neural networks assuming infinite memory, while chip designers built hardware to run generic code. But Moore's Law and Dennard scaling have slowed: processors cannot simply get faster and cooler by packing more transistors. Today, big speedups only happen when models are shaped around hardware strengths—like matching layer sizes to hardware cache lines or structuring sparsity so GPU tensor cores can skip empty values.

## Minimal example
In early AI pruning, researchers randomly deleted individual zero-weights everywhere. While the model had 90% fewer numbers on paper, GPUs ran it *slower* than before because random memory lookups caused cache misses. In hardware-aware pruning, researchers enforce "2:4 structured sparsity" (every 4 numbers must contain at least 2 zeros). NVIDIA GPUs added hardware support for this exact pattern, cutting compute and memory traffic in half with zero lookup overhead.

## Prerequisites
- [[concepts/the-memory-compute-gap|The Memory-Compute Gap]]

## Common confusions
- Thinking you always have to build a new chip: Most co-design is pure software. It means writing models and inference engines that understand how existing hardware caches, memory alignments, and SIMD registers actually work.

## Appears in
- [[lessons/2026-11-01-intro-to-tinyml-and-efficient-ai]]: presented as the primary driver of modern AI speedups after the slowdown of Moore's Law.
