---
title: Activation Memory vs Weight Memory
aliases: [activation memory, weight memory, peak memory, memory footprint]
tags: [concept, computer-systems, efficiency]
---

## Definition
**Weight memory** is the static storage required to hold a model's trained parameters, while **activation memory** is the dynamic memory needed during inference or training to hold intermediate feature maps and layer outputs.

## Why it's true
Developers often focus only on model size (weight memory), assuming a smaller model will always fit on a small device. But in vision models and edge devices, **peak activation memory is often the true bottleneck**. In early layers, high image resolution ($H \times W$) creates massive activation tensors; in later layers, high channel counts ($C_i \times C_o$) create large weight tensors. A model with very few weights can still crash a device if an intermediate layer creates a giant activation tensor.

## Minimal example
Comparing MobileNetV2 to ResNet-50:
- MobileNetV2 reduced weight parameters by **4.3×** (from 102 MB down to 25 MB).
- However, its **peak activation memory actually increased by 1.8×**!
Because MobileNetV2 expands channels by 6× before depthwise convolutions, its intermediate activation tensors grew significantly larger, creating an unexpected memory spike on memory-constrained microcontrollers.

## Prerequisites
- [[concepts/the-memory-compute-gap|The Memory-Compute Gap]]

## Common confusions
- Conflating model parameter size on disk with runtime memory usage: A 25 MB model can easily require 200 MB of RAM at runtime because it must hold input buffers and peak activation tensors in memory simultaneously.

## Appears in
- [[lessons/2026-11-01-basics-of-neural-networks-and-efficiency-metrics]]: explains why activation memory is frequently the real barrier to deploying models on edge devices.
