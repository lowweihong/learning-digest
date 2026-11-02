---
title: Linear Quantization
aliases: [uniform quantization, affine quantization, integer quantization, INT8 quantization]
tags: [concept, quantization, hardware-acceleration]
---

## Definition
A mathematical ruler that maps decimal numbers $r$ into whole numbers (integers) $q$ using the equation $r = S(q - Z)$, where $S$ is the step size (scale) and $Z$ is the integer zero-point.

## Why it's true
Cheap chips (like microcontrollers in IoT devices and smart appliances) do not have floating-point units for decimal math. Because matrix multiplication and convolution are linear, using $r = S(q - Z)$ lets processors do all inner loop math with fast, low-power integer multiply-accumulate (MAC) instructions, scaling the result at the end with a simple bit-shift.

## Minimal example
Suppose numbers range from $-1.08$ to $+2.12$, and you want to fit them into 2-bit integers ($-2, -1, 0, 1$):
1. **Step size ($S$):** Divide the decimal spread by the integer spread:
   $$S = \frac{2.12 - (-1.08)}{1 - (-2)} = \frac{3.20}{3} \approx 1.07$$
2. **Zero-Point ($Z$):** Align continuous zero to an exact integer:
   $$Z = \text{round}\left(-2 - \frac{-1.08}{1.07}\right) = -1$$
3. **Quantizing a weight ($r = 1.49$):**
   $$q = \text{round}\left(\frac{1.49}{1.07}\right) + (-1) = 1 - 1 = 0$$
4. **Reconstructed:** $\hat{r} = 1.07 \times (0 - (-1)) = 1.07$.

## Prerequisites
- [[concepts/floating-point-representation|Floating-Point Representation]]
- [[concepts/zero-point|Zero-Point]]

## Common confusions
- Assuming you need decimal math to scale the output: In silicon, the scaling factor $M = \frac{S_w S_x}{S_y}$ is turned into a fixed integer multiplication followed by an arithmetic bit-shift to the right, avoiding all decimal hardware.

## Appears in
- [[lessons/2026-11-01-quantization-k-means-and-linear]]: provides the mathematical derivation for integer-only GEMM and convolution.
