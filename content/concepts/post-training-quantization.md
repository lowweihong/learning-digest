---
title: Post-Training Quantization
aliases: [PTQ, offline quantization, zero-shot quantization]
tags: [concept, model-compression, quantization, deep-learning]
---

## Definition
A quantization workflow that converts a pre-trained floating-point neural network into integer precision (such as INT8 or FP4) using a small unlabeled calibration dataset without requiring full model retraining or backpropagation.

## Why it's true
Retraining massive models is expensive, slow, and often impossible when training data is proprietary. Because well-trained neural network weights occupy relatively smooth distributions, their dynamic range can be calculated offline, and activation statistics can be captured by running a few dozen calibration batches through the network. Applying per-channel scaling, outlier clipping (via KL divergence), and intelligent rounding (like AdaRound) enables PTQ to recover near-baseline accuracy in minutes.

## Minimal example
Quantizing a 16-bit floating-point ResNet to 8-bit integers:
1. **Weight Calibration:** Find $R_{\max} = \max(|W|)$ per output channel to calculate individual scale factors $S_c = R_{\max} / 127$.
2. **Activation Calibration:** Feed 100 validation images forward through the model to record activation distributions after each layer, choosing clipping thresholds that minimize information loss (KL divergence).
3. **Integer Export:** Quantize weights and precompute activation scale factors. The model shrinks by $4\times$ and runs entirely in INT8 arithmetic on mobile or GPU integer units with $<0.5\%$ accuracy loss.

## Prerequisites
- [[concepts/linear-quantization|Linear Quantization]]

## Common confusions
- **Per-tensor vs. Per-channel PTQ:** Applying a single scale factor to an entire tensor fails when different channels have wildly varying magnitudes (e.g., in LLMs where outlier channels are $100\times$ larger than normal channels). Per-channel quantization solves this by assigning an independent scale factor $S_c$ to each output channel.

## Appears in
- [[lessons/2026-11-01-advanced-quantization-and-extreme-compression]]: introduced as the primary fast-deployment quantization technique.
