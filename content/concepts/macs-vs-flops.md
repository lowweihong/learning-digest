---
title: MACs vs FLOPs
aliases: [MAC, FLOP, multiply-accumulate, floating-point operations]
tags: [concept, computer-systems, efficiency]
---

## Definition
A Multiply-Accumulate (MAC) is an operation that computes $a \leftarrow a + (b \times c)$, consisting of exactly one multiplication and one addition, which equals **2 Floating-Point Operations (FLOPs)**.

## Why it's true
Most neural network calculations (like matrix multiplication and convolution) are dot products: you multiply a weight by an input and add it to a running total. Because silicon processors often have dedicated hardware units that do this multiply-add step in a single clock cycle, hardware counts them as 1 MAC. In theoretical math, because there are two distinct operations (one multiply, one add), it counts as 2 FLOPs.

## Minimal example
If an image classification model (like AlexNet) requires **724 million MACs** to process a picture:
$$\text{FLOPs} = 724 \times 10^6 \times 2 = 1.45 \text{ GigaFLOPs (GFLOPs)}$$
If an accelerator runs at 1.45 GFLOPs per second, it will process roughly one image per second.

## Prerequisites
- None

## Common confusions
- Confusing lowercase FLOP (floating-point operation, a count of work) with uppercase FLOPS (floating-point operations per second, a measure of hardware speed).

## Appears in
- [[lessons/2026-11-01-basics-of-neural-networks-and-efficiency-metrics]]: explains how to calculate computational workload in deep learning.
