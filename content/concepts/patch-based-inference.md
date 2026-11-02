---
title: Patch-Based Inference
aliases: [patch-by-patch inference, tiled activation execution, per-patch execution]
tags: [concept, tinyml, efficient-ai, computer-vision, systems]
---

## Definition
An inference execution schedule that partitions high-resolution spatial feature maps into smaller geometric tiles (patches) and computes multiple consecutive early neural network layers patch-by-patch, eliminating the need to store entire high-resolution intermediate activation tensors in SRAM.

## Why it's true
In standard **per-layer execution**, a deep neural network executes layer-by-layer across the whole tensor:
$$\text{Layer } 1(X_{\text{full}}) \to H_1 \implies \text{Layer } 2(H_1) \to H_2 \implies \dots$$
Because the entire intermediate activation tensor $H_1$ must be materialized in SRAM before Layer 2 can begin, memory consumption peaks at the very beginning of the network where spatial resolutions ($H \times W$) are highest. On microcontrollers with only $256\text{ KB}$ of SRAM, this activation memory spike creates a hard barrier that prevents running modern vision models.

**Patch-Based Inference** reorganizes the execution loop:
1. **Spatial Tiling:** The input image is divided into a grid of spatial patches (e.g., $2 \times 2$ or $4 \times 4$).
2. **Depth-First Tile Processing:** A single spatial patch is propagated through several early convolutional layers in sequence, producing an output patch whose resolution is progressively downsampled by stride operations.
3. **Low-Memory Aggregation:** Once the spatial dimensions are sufficiently compressed by downsampling layers, the patches are stitched back together in SRAM, and the remainder of the network transitions back to standard per-layer execution.

By holding only a fraction of the intermediate activation map in SRAM at any given moment, peak memory usage is reduced by up to $8\times$, allowing microcontrollers to process higher-resolution inputs (e.g., $160 \times 160$ vs. $128 \times 128$) without increasing silicon memory.

## Minimal example
Consider an initial convolutional layer processing an input of $160 \times 160 \times 3$ that outputs $160 \times 160 \times 16$ with 8-bit integer activations:
- **Per-Layer Execution:**
  $$\text{SRAM Required} = \text{Input} + \text{Output} = (160 \times 160 \times 3) + (160 \times 160 \times 16) = 76.8\text{ KB} + 409.6\text{ KB} = 486.4\text{ KB}$$
  This immediately crashes a $256\text{ KB}$ or $320\text{ KB}$ microcontroller.
- **Patch-Based Execution ($2 \times 2$ tiling):**
  Each patch is approximately $80 \times 80$:
  $$\text{SRAM Required} \approx (80 \times 80 \times 3) + (80 \times 80 \times 16) \approx 19.2\text{ KB} + 102.4\text{ KB} = 121.6\text{ KB}$$
  The layer executes comfortably within a $256\text{ KB}$ SRAM budget with memory to spare.

## Prerequisites
- [[concepts/activation-memory-vs-weight-memory|Activation Memory vs Weight Memory]], [[concepts/mcunet|MCUNet]]

## Common confusions
- **Patch inference alters mathematical outputs:** When boundary padding (the halo region) is properly accounted for, patch-based inference produces the exact same numerical outputs as standard per-layer convolution. It changes only the execution order and memory lifetime, not the mathematical function.
- **Applying patches across all layers:** Patch-based execution is only beneficial in early layers where spatial resolution is large. In deep layers, channels are wide and spatial maps are tiny (e.g., $7 \times 7$), so per-layer execution is far more computationally efficient.

## Appears in
- [[lessons/2026-11-01-tinyml-mcunet-and-patch-inference]]: introduced as the key execution breakthrough in MCUNetV2 that breaks the SRAM activation bottleneck.
