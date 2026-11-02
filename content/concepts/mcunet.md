---
title: MCUNet
aliases: [MCUNetV1, TinyEngine and TinyNAS, microcontroller deep learning]
tags: [concept, tinyml, efficient-ai, systems, deep-learning]
---

## Definition
A unified system-algorithm co-design framework that enables ImageNet-scale deep learning on off-the-shelf microcontrollers (with a few hundred kilobytes of SRAM and 1–2 megabytes of Flash) by jointly optimizing neural network architecture search (**TinyNAS**) and bare-metal runtime compilation (**TinyEngine**).

## Why it's true
Microcontrollers are fundamentally different from smartphones and cloud GPUs. They lack operating systems, virtual memory, external DRAM, and dynamic garbage collectors. Deploying neural networks onto microcontrollers requires satisfying two brutal, non-negotiable physical ceilings:
1. **Flash Storage Ceiling ($\sim 1\text{ MB}$):** Non-volatile memory that stores the compiled firmware and static neural network weights.
2. **SRAM Computation Ceiling ($\sim 256\text{–}320\text{ KB}$):** Ultra-fast on-chip memory that must hold the runtime stack, input activations, and output activations for every layer.

Standard mobile architectures (such as MobileNetV2) and off-the-shelf embedded runtimes (like TensorFlow Lite Micro) fail because they are designed in isolation:
- MobileNetV2 optimizes for mobile phones with gigabytes of RAM; its earliest layers generate activation tensors demanding $>1,300\text{ KB}$ of SRAM—causing immediate memory overflow on microcontrollers.
- Standard interpreters allocate runtime memory dynamically and carry heavy framework overhead.

MCUNet solves this through two symbiotic pillars:
- **TinyNAS:** Employs two-stage architecture search to explore non-uniform kernel sizes, expansion ratios, and input resolutions that flatten the peak activation profile across layers, preventing early-layer memory bottlenecks.
- **TinyEngine:** A bare-metal compiler that performs full static memory planning, generates direct C code with no interpretation overhead, and schedules in-place tensor reuse buffers.

## Minimal example
Deploying an ImageNet vision model onto an STM32F746 microcontroller ($320\text{ KB}$ SRAM, $1\text{ MB}$ Flash):
- **TensorFlow Lite Micro + MobileNetV2-0.35:** Fails with an Out-Of-Memory (OOM) hard fault during layer 1 because peak activation memory requires $1,340\text{ KB}$ SRAM ($4.2\times$ over device capacity).
- **MCUNet:** TinyNAS discovers an architecture whose peak activation memory never exceeds $256\text{ KB}$, while TinyEngine compiles it with zero runtime memory fragmentation. The system achieves $70.7\%$ ImageNet top-1 accuracy in real-time on bare-metal silicon.

## Prerequisites
- [[concepts/tinyml|TinyML]], [[concepts/tinynas|TinyNAS]], [[concepts/algorithm-hardware-co-design|Algorithm-Hardware Co-Design]]

## Common confusions
- **MCUNet is just a lightweight CNN:** MCUNet is not a single static neural network; it is a co-design methodology pairing an automated search space optimizer (TinyNAS) with an ultra-lean static compiler (TinyEngine).
- **Flash is the primary bottleneck:** While fitting weights into $1\text{ MB}$ Flash requires quantization and pruning, the far harder bottleneck on microcontrollers is almost always **peak SRAM activation memory**, which is $4\times$ to $5\times$ smaller than Flash.

## Appears in
- [[lessons/2026-11-01-tinyml-mcunet-and-patch-inference]]: introduced as the foundational system-algorithm co-design framework bringing deep learning to microcontrollers.
