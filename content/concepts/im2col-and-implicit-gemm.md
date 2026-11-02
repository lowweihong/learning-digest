---
title: Im2Col and Implicit GEMM
aliases: [image-to-column, implicit GEMM, convolution as matrix multiplication, GEMM-based convolution]
tags: [concept, systems, deep-learning, computer-vision, efficient-ai]
---

## Definition
**Im2Col (Image to Column)** is an algorithmic transformation that unrolls overlapping spatial sliding windows of an input feature map into columns of a 2D matrix, allowing convolution to be computed using highly optimized General Matrix Multiplication (GEMM) libraries. **Implicit GEMM** performs this conversion virtually inside compute registers, avoiding the massive memory inflation of explicit Im2Col matrices.

## Why it's true
Modern hardware architectures (CPUs, GPUs, TPUs) are designed to execute General Matrix Multiply (GEMM) at peak silicon efficiency, utilizing hardware Tensor Cores, SIMD vector units, and vendor BLAS routines (like cuBLAS or oneMKL). Convolutions, however, are complex multi-dimensional sliding windows with non-trivial boundary and striding patterns:
$$\text{Output}(c_{\text{out}}, y, x) = \sum_{c_{\text{in}}} \sum_{ky} \sum_{kx} W(c_{\text{out}}, c_{\text{in}}, ky, kx) \cdot \text{Input}(c_{\text{in}}, y + ky, x + kx)$$

Im2Col maps this operation directly onto GEMM ($C = A \cdot B$):
1. **Weight Matrix ($A$):** Reshape convolutional filters of shape $(C_{\text{out}}, C_{\text{in}}, K, K)$ into a 2D matrix of shape $C_{\text{out}} \times (C_{\text{in}} \cdot K \cdot K)$.
2. **Feature Matrix ($B$):** For every output pixel $(y, x)$, extract the $K \times K \times C_{\text{in}}$ receptive field patch and flatten it into a single column of shape $(C_{\text{in}} \cdot K \cdot K) \times (H_{\text{out}} \cdot W_{\text{out}})$.
3. **GEMM Execution:** Multiply $A \times B$. The resulting matrix has shape $C_{\text{out}} \times (H_{\text{out}} \cdot W_{\text{out}})$, which matches the desired output feature map when reshaped.

#### The Im2Col Memory Dilemma and Implicit GEMM
Because adjacent sliding windows overlap substantially, explicit Im2Col duplicates each input pixel up to $K \times K$ times (e.g., $9\times$ for $3 \times 3$ kernels). On an image of size $224 \times 224 \times 64$, the Im2Col buffer explodes to over $115\text{ MB}$—an impossible overhead for mobile chips and microcontrollers.

**Implicit GEMM (Fused GEMM)** eliminates this buffer:
- The GPU or CPU kernel executes the standard GEMM loop.
- When an inner product requires element $B[k][j]$, the thread computes the corresponding $(c_{\text{in}}, y + ky, x + kx)$ coordinate on-the-fly using integer division and modulo arithmetic.
- Data is fetched directly from the original compact image tensor into local registers, achieving $100\%$ of GEMM hardware performance with zero intermediate memory allocation.

## Minimal example
Executing a $3 \times 3$ convolution with 64 input channels and 128 output channels on a $112 \times 112$ feature map:
- **Input Tensor Size:** $112 \times 112 \times 64 \times 4\text{ bytes} \approx 3.21\text{ MB}$.
- **Explicit Im2Col Buffer:** $12,544\text{ patches} \times (3 \times 3 \times 64) \times 4\text{ bytes} \approx 28.9\text{ MB}$ ($9\times$ memory bloat).
- **Implicit GEMM Execution:** cuDNN or CUTLASS executes a fused GEMM kernel. Threads calculate pixel offsets mathematically in registers and stream data directly into matrix multiplication units. Peak memory remains strictly $3.21\text{ MB}$.

## Prerequisites
- [[concepts/the-memory-compute-gap|The Memory-Compute Gap]], [[concepts/macs-vs-flops|MACs vs FLOPs]]

## Common confusions
- **Im2Col changes the mathematical convolution output:** Im2Col and Implicit GEMM compute the exact same mathematical convolution; they only reorganize arithmetic ordering to leverage GEMM acceleration.
- **Implicit GEMM is slower due to coordinate math:** On modern GPUs and CPUs, index calculations (shifts, bitmasks, modulo) are pipelined concurrently in integer ALUs while floating-point units execute FMA operations, meaning index calculation overhead is completely hidden behind memory access latency.

## Appears in
- [[lessons/2026-11-01-tinyengine-parallel-processing-and-inference-optimizations]]: introduced as the industry-standard mechanism for mapping convolution operations to GEMM hardware.
