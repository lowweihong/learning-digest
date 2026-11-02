---
title: Depthwise Separable Convolution
aliases: [depthwise convolution, separable convolution, DW-Conv]
tags: [concept, neural-networks, efficiency]
---

## Definition
A lightweight convolution technique that splits a standard 3D convolution into two separate steps: a **depthwise convolution** (filtering each input channel individually) followed by a **pointwise convolution** (a $1 \times 1$ convolution combining information across channels).

## Why it's true
In a standard convolution with a $3 \times 3$ kernel, every filter simultaneously looks across all input channels and across spatial pixels ($3 \times 3 \times C_{\text{in}} \times C_{\text{out}}$). By decoupling spatial filtering from cross-channel mixing, depthwise separable convolution reduces computation and parameter counts by roughly $\frac{1}{C_{\text{out}}} + \frac{1}{K^2}$ (nearly an **8× to 9× reduction** for $3 \times 3$ kernels) with minimal drop in accuracy.

## Minimal example
Suppose you have an input with 64 channels, outputting 64 channels with a $3 \times 3$ kernel:
- **Standard Convolution:** Needs $3 \times 3 \times 64 \times 64 = 36,864$ operations per output pixel.
- **Depthwise Separable Convolution:**
  1. Depthwise step ($3 \times 3$ per channel): $3 \times 3 \times 64 = 576$ operations.
  2. Pointwise step ($1 \times 1$ across channels): $1 \times 1 \times 64 \times 64 = 4,096$ operations.
  3. Total: $576 + 4,096 = 4,672$ operations—**almost 8× fewer calculations**!

## Prerequisites
- [[concepts/macs-vs-flops|MACs vs FLOPs]]

## Common confusions
- Assuming depthwise convolution replaces all convolutions: A depthwise convolution on its own cannot mix information between channels; it must always be paired with a $1 \times 1$ pointwise convolution to allow channels to communicate.

## Appears in
- [[lessons/2026-11-01-basics-of-neural-networks-and-efficiency-metrics]]: introduced as the key architectural building block of MobileNet and modern efficient vision models.
