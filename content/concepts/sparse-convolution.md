---
title: Sparse Convolution
aliases: [submanifold sparse convolution, Minkowski convolution]
tags: [concept, deep-learning, hardware, robotics]
---

## Definition
A specialized convolution operation for spatially sparse data (such as 3D LiDAR point clouds or medical imaging) that restricts output computations strictly to occupied spatial coordinates, preventing sparse inputs from dilating and becoming dense in deeper layers.

## Why it's true
In physical 3D space, LiDAR point clouds collected by autonomous vehicles or robotics are over $95\%$ to $99\%$ empty space (free air). If processed with standard 3D convolutions, every $3 \times 3 \times 3$ kernel dilates non-zero activations to adjacent empty voxels. Within three to four convolutional layers, the entire 3D volume becomes completely dense, causing memory usage and arithmetic compute to explode.

**Submanifold Sparse Convolution** solves this by enforcing an invariant: an output feature vector is calculated *if and only if* the center coordinate was already occupied in the input. This maintains constant spatial sparsity across all depths of the network.

## Minimal example
Consider a 1D sparse signal with non-zero points at indices $x = [2, 7]$ processed by a 3-element kernel $[-1, 0, 1]$:
- **Standard convolution:** Outputs non-zero values at $[1, 2, 3]$ and $[6, 7, 8]$. The signal expands from 2 occupied voxels to 6 occupied voxels (densification).
- **Submanifold sparse convolution:** Computes dot products only for the valid occupied center indices $x = 2$ and $x = 7$. Unoccupied positions remain exact zeros with zero memory allocation.

Systems like **TorchSparse** and hardware like **PointAcc** optimize this through adaptive grouping, clustering sparse coordinates with similar arithmetic density into regular GPU thread blocks to avoid warp starvation.

## Prerequisites
- [[concepts/static-vs-dynamic-sparsity|Static vs Dynamic Sparsity]]

## Common confusions
- **Regular vs. Submanifold Sparse Convolution:** Regular sparse convolution allows non-zero outputs wherever the kernel overlaps at least one active input voxel (which still dilates space slowly). Submanifold sparse convolution strictly requires the output center position itself to be non-zero, guaranteeing zero spatial dilation.

## Appears in
- [[lessons/2026-11-01-automated-pruning-and-hardware-sparsity]]: introduced as the software (TorchSparse) and hardware (PointAcc) solution for spatial activation sparsity in point clouds.
