---
title: Inverted Residual Block
aliases: [inverted bottleneck, MBConv, MobileNetV2 block]
tags: [concept, deep-learning, efficient-ai, cnn]
---

## Definition
A convolutional building block that expands low-dimensional input channels into a high-dimensional feature space using a $1 \times 1$ pointwise convolution, executes an efficient $3 \times 3$ depthwise convolution in the expanded space, and projects back down to a low-dimensional bottleneck with a linear $1 \times 1$ convolution.

## Why it's true
In classic ResNet bottleneck blocks, high-dimensional channels ($2048$) are first compressed into a low-dimensional bottleneck ($512$), processed by a standard convolution, and then expanded back to high dimensions. 

MobileNetV2 reverses this pattern because **depthwise convolutions are computationally cheap but have low expressive capacity**. By expanding channels by $6\times$ (e.g., from $N$ to $6N$) before the depthwise convolution, the cheap spatial filter operates on rich, high-dimensional features without exploding arithmetic compute. The final projection uses a linear (non-ReLU) bottleneck to prevent non-linearities from zeroing out negative information in low-dimensional manifolds.

## Minimal example
In a layer transforming 160 input channels:
1. **$1 \times 1$ Expansion:** Multiplies channels by $6\times$ ($160 \to 960$).
2. **$3 \times 3$ Depthwise Convolution:** Convolves spatially on each of the 960 channels independently (mapping 1:1, requiring only $960 \times 9$ parameters).
3. **$1 \times 1$ Linear Bottleneck:** Projects $960$ channels back down to $160$ output channels without a non-linear activation.
4. **Residual Connection:** Directly adds the 160 input channels to the 160 output channels.

While parameters shrink by $4.6\times$ compared to ResNet-18, the temporary $6\times$ channel expansion causes intermediate activation memory to expand by up to $80\%$, creating a dynamic memory bottleneck on embedded RAM.

## Prerequisites
- [[concepts/depthwise-separable-convolution|Depthwise Separable Convolution]]

## Common confusions
- **Parameter efficiency $\neq$ Activation efficiency:** Expanding channels by $6\times$ in the middle layer makes the block lightweight in weights, but dramatically inflates peak activation memory ($H \times W \times 6N$). On microcontrollers with tiny SRAM, inverted bottlenecks often run out of memory unless specifically scaled by algorithms like TinyNAS.

## Appears in
- [[lessons/2026-11-01-neural-architecture-search-foundations]]: analyzed as the ubiquitous mobile building block and its hidden activation memory penalty.
