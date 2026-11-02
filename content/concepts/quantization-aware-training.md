---
title: Quantization-Aware Training
aliases: [QAT, simulated quantization, fake quantization]
tags: [concept, model-compression, quantization, training]
---

## Definition
A training technique that models the mathematical noise and precision clamping of low-precision integer arithmetic during the forward pass, using gradient descent to allow the network to adapt and recover accuracy before deployment.

## Why it's true
Compact architectures (such as MobileNet) and aggressive low-bit targets (such as 4-bit weights and activations) suffer severe accuracy collapse under post-training quantization because the model has no redundant parameters to absorb rounding errors. 

QAT resolves this by inserting "fake quantization" nodes into the computational graph. The network computes forward activations using clamped, discrete numbers, exposing the model to the exact rounding errors it will face on hardware, while backpropagation adjusts weights to find loss basins that are naturally resilient to precision loss.

## Minimal example
On MobileNet-v2:
1. **PTQ baseline:** Quantizing straight to INT8 without retraining drops top-1 accuracy from $71.8\%$ down to a catastrophic $0.1\%$.
2. **QAT execution:**
   - Maintain a master copy of weights in full-precision FP32.
   - In the forward pass, clamp and round weights and activations to simulated INT8 values: $\tilde{w} = S \cdot \text{round}(w / S)$.
   - Run forward operations and compute task loss.
   - Backpropagate gradients to update the FP32 master weights.
   - After a few epochs of fine-tuning, accuracy fully recovers from $0.1\%$ to $70.4\%$.

## Prerequisites
- [[concepts/linear-quantization|Linear Quantization]]

## Common confusions
- **Physical vs. Simulated low precision during QAT:** Hardware during QAT does not run on integer registers. The weights and activations are stored and multiplied as standard floating-point numbers (FP32), but their numerical values are strictly constrained to representable integer grid points. Actual integer conversion happens only at final export.

## Appears in
- [[lessons/2026-11-01-advanced-quantization-and-extreme-compression]]: introduced as the critical recovery mechanism when PTQ causes unacceptable accuracy loss.
