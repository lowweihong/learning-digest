---
title: In-Place Depthwise Convolution
aliases: [in-place convolution, memory-efficient depthwise convolution, TinyEngine buffer optimization]
tags: [concept, tinyml, efficient-ai, systems, computer-vision]
---

## Definition
A memory optimization technique implemented in microcontroller inference engines (such as **TinyEngine**) that computes depthwise convolutions by allocating only a single temporary 2D channel buffer, overwriting the input channel memory in-place once each channel's spatial convolution completes, reducing peak SRAM from $2 \cdot C \cdot H \cdot W$ down to $(1 + C) \cdot H \cdot W$.

## Why it's true
In depthwise separable convolutions (the core building block of MobileNetV2 and MCUNet inverted residual blocks), depthwise layers apply an independent spatial filter to each channel:
$$\text{Output}(c, y, x) = \sum_{ky} \sum_{kx} W(c, ky, kx) \cdot \text{Input}(c, y + ky, x + kx)$$
Notice that computing channel $c$ of the output requires **only channel $c$ of the input**. There is zero mathematical cross-talk or information exchange across different channels.

Standard embedded inference runtimes (such as TensorFlow Lite Micro) treat depthwise convolution as a generic layer and allocate two complete, separate 3D tensors in SRAM simultaneously:
$$\text{Peak SRAM} = \text{Input Buffer} + \text{Output Buffer} = 2 \cdot C \cdot H \cdot W$$
Inside an inverted residual block where channel width is expanded by $6\times$, holding both tensors simultaneously creates a massive memory spike that exceeds microcontroller SRAM limits.

**In-Place Depthwise Convolution** leverages channel independence:
1. **Allocate a 1-Channel Buffer:** Allocate a tiny scratchpad buffer of size $1 \times H \times W$.
2. **Channel-by-Channel Execution:** Compute the spatial convolution for channel $0$ into the scratchpad buffer.
3. **In-Place Overwrite:** Because channel $0$ input is never needed again by any subsequent channel, copy the scratchpad output directly into the memory region originally occupied by channel $0$ input.
4. **Repeat for Channels $1 \dots C-1$:** Reuse the exact same scratchpad buffer for each subsequent channel.

The peak SRAM allocation shrinks from $2CHW$ to $(C + 1)HW$—a near **$50\%$ reduction in peak activation SRAM** with zero loss of numerical fidelity.

## Minimal example
A depthwise convolution layer with $C = 96$ channels and spatial size $28 \times 28$ running on INT8 activations (1 byte per element):
- **Standard Allocation:**
  $$\text{Memory} = 2 \times (96 \times 28 \times 28 \times 1\text{ byte}) = 2 \times 75.26\text{ KB} = 150.5\text{ KB}$$
  This consumes over half of a $256\text{ KB}$ microcontroller's SRAM.
- **In-Place Execution (TinyEngine):**
  $$\text{Memory} = (96 + 1) \times (28 \times 28 \times 1\text{ byte}) = 97 \times 784\text{ bytes} \approx 76.05\text{ KB}$$
  The layer saves $74.5\text{ KB}$ of SRAM ($49.5\%$ reduction), allowing the remaining SRAM to be used for larger batch buffers or deeper model capacity.

## Prerequisites
- [[concepts/depthwise-separable-convolution|Depthwise Separable Convolution]], [[concepts/tinyml|TinyML]]

## Common confusions
- **Can standard convolutions run in-place?** No. In standard (dense) convolutions, every output channel depends on *all* input channels. Overwriting input channel 0 would destroy data needed by output channels 1 through $C_{\text{out}}-1$. In-place execution is uniquely possible for depthwise convolutions because channels are completely decoupled.
- **In-place computation slows down execution:** Copying an $H \times W$ channel slice inside fast L1 SRAM or registers takes negligible clock cycles compared to the arithmetic execution of the 2D convolution, resulting in zero measurable latency penalty.

## Appears in
- [[lessons/2026-11-01-tinyengine-parallel-processing-and-inference-optimizations]]: introduced as the key memory-saving operator optimization in TinyEngine.
