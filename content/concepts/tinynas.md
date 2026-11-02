---
title: TinyNAS
aliases: [MCUNet NAS, microcontroller search space optimization]
tags: [concept, tinyml, automl, hardware, efficient-ai]
---

## Definition
A two-stage neural architecture search framework specifically designed for extreme microcontroller memory constraints (e.g., $256\text{ KB}$ SRAM), which combines automated search space optimization with memory-bounded model specialization to maximize arithmetic compute under strict SRAM and Flash limits.

## Why it's true
Standard mobile NAS search spaces (like MobileNetV2 or ProxylessNAS) were designed for smartphones with gigabytes of RAM. When deployed to microcontrollers with only $256\text{–}320\text{ KB}$ of SRAM, over $99\%$ of candidate networks crash with out-of-memory errors on the very first residual block.

TinyNAS resolves this through a core physical principle: **Computation is cheap, but memory is expensive.** Under a hard memory ceiling (e.g., peak activation $\le 256\text{ KB}$ and model size $\le 1\text{ MB}$), a search space that produces higher cumulative FLOPs provides higher representational capacity and strictly superior task accuracy. TinyNAS automatically prunes away unpromising or memory-violating sub-spaces before exploration begins, optimizing the distribution of FLOPs to produce models that achieve over $70\%$ ImageNet accuracy on off-the-shelf Cortex-M7 microcontrollers (MCUNet).

## Minimal example
Designing vision networks for a $320\text{ KB}$ SRAM microcontroller:
1. **Search Space Optimization:** Evaluate candidate search spaces by plotting their cumulative FLOP distributions under the $320\text{ KB}$ SRAM constraint. Discard search spaces with low average FLOPs (like standard ResNet-18 variants, which run out of memory).
2. **Model Specialization:** Search for non-uniform kernel sizes, expansion ratios, and resolution scales that pack maximum compute into the $320\text{ KB}$ SRAM window.
3. **Outcome:** Generates MCUNet models that deliver $70.7\%$ ImageNet accuracy using just $256\text{ KB}$ SRAM and $900\text{ KB}$ Flash—a task previously considered impossible on microcontrollers without external DRAM.

## Prerequisites
- [[concepts/neural-architecture-search|Neural Architecture Search]]
- [[concepts/tinyml|TinyML]]

## Common confusions
- **Minimizing FLOPs vs. Maximizing FLOPs under memory limits:** For cloud and battery-constrained mobile phones, NAS aims to *minimize* FLOPs for a given accuracy. On microcontrollers, memory (SRAM) is the hard brick wall, not compute. TinyNAS *maximizes* FLOPs subject to fitting inside SRAM, because higher FLOPs within the memory envelope directly maximizes accuracy.

## Appears in
- [[lessons/2026-11-01-neural-architecture-search-foundations]]: introduced as the search space optimization technique that brought deep learning to microcontrollers.
