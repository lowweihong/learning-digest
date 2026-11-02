---
title: Micro-Scaling Formats
aliases: [NVFP4, MX formats, group quantization, microscaling]
tags: [concept, hardware, quantization, deep-learning]
---

## Definition
A family of sub-byte floating-point formats (such as NVIDIA Blackwell NVFP4 and OCP Microscaling MX4/MX6/MX9) that group small vectors of numbers (e.g., 2, 16, or 32 elements) to share a common scaling exponent, achieving ultra-low-bit precision with minimal area and quantization noise.

## Why it's true
Pushing global quantization below 8 bits (e.g. to FP4 or INT4) encounters steep accuracy degradation because a single layer-wide or channel-wide scaling factor cannot capture local dynamic range variations.

Micro-scaling introduces a **multi-level scaling hierarchy**:
- **L0 (Fine-Grained Local Scale):** A tiny local exponent or scale factor shared across a compact vector (e.g., a 4-bit scale for every 16 elements in VS-Quant, or a 1-bit exponent for every 2 elements in MX4).
- **L1 (Global Scale):** A standard FP16 or FP32 tensor-level scale factor that captures overarching layer magnitude.

By sharing exponents over tiny local groups, the format dampens local outlier spikes without inflating memory overhead: storing a 4-bit scale per 16 elements adds only $\frac{4}{16} = 0.25$ bits per weight, yielding an effective bit-width of $4.25$ bits.

## Minimal example
In **NVIDIA Blackwell NVFP4**:
- Tensor Cores achieve **18 PFLOPS** of peak math throughput ($2\times$ faster than FP8 and $4\times$ faster than FP16).
- Every micro-tensor block shares an 8-bit scale factor across a small vector.
- Instead of computing floating-point multiplications on full 32-bit registers, hardware multiplies compact 4-bit numbers directly and scales their accumulated partial sums once per micro-block, slashing silicon area and DRAM bandwidth simultaneously.

In **OCP MX4 (Microscaling 4-bit)**:
- 3 bits per element (1 sign bit + 2 mantissa bits).
- 1 exponent bit shared across every 2 elements ($+0.5$ bits).
- 8-bit scale shared across 16 elements ($+0.5$ bits).
- Total effective footprint: $3 + 0.5 + 0.5 = 4.0$ bits per weight.

## Prerequisites
- [[concepts/post-training-quantization|Post-Training Quantization]]

## Common confusions
- **Group quantization vs. Per-channel quantization:** Per-channel quantization shares one scale factor across hundreds of weights along a matrix column ($C_{\text{in}} \times K \times K$). Group/micro-scaling quantizes small fixed-size vectors (e.g., 16 or 32 adjacent numbers) independently of channel boundaries, providing much tighter local bounds against activation outliers.

## Appears in
- [[lessons/2026-11-01-advanced-quantization-and-extreme-compression]]: introduced as modern silicon's answer to scaling FP4 and sub-byte precision without accuracy collapse.
