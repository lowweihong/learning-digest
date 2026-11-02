---
title: Winograd Convolution
aliases: [Winograd fast convolution, Winograd algorithm, F(2x2, 3x3), minimal filtering convolution]
tags: [concept, deep-learning, efficient-ai, computer-vision, systems]
---

## Definition
A fast mathematical convolution algorithm that drastically reduces the number of expensive multiplication operations required for small convolutional filters (such as $3 \times 3$) by transforming the input image and kernel weights into a minimal filtering domain, computing element-wise matrix multiplications, and transforming the result back into spatial outputs.

## Why it's true
In digital hardware design, multiplication operations consume dramatically more silicon area, clock latency, and electrical energy than additions or bitwise shifts. Standard direct convolution requires $K^2$ multiplications for every single output pixel. For a $3 \times 3$ kernel producing a $2 \times 2$ output block:
$$\text{Standard Multiplications} = 2 \times 2 \times 3 \times 3 = 36\text{ multiplications}$$

The Winograd minimal filtering algorithm—specifically $F(m \times m, r \times r)$, which computes an $m \times m$ output tile using an $r \times r$ kernel from an $(m + r - 1) \times (m + r - 1)$ input tile—proves that a $2 \times 2$ output with a $3 \times 3$ filter can be computed with only 16 multiplications:
$$Y = A^T \left[ (G g G^T) \odot (B^T d B) \right] A$$
where:
1. **$g$ is the $3 \times 3$ filter**, transformed offline into a $4 \times 4$ representation by matrix $G$: $U = G g G^T$.
2. **$d$ is the $4 \times 4$ input patch**, transformed at runtime into a $4 \times 4$ representation by matrix $B$: $V = B^T d B$.
3. **$\odot$ denotes element-wise Hadamard multiplication**, requiring exactly $4 \times 4 = 16$ multiplications.
4. **$A$ is the inverse transformation matrix**, converting the $4 \times 4$ intermediate result back to the $2 \times 2$ spatial output tile $Y$.

Because the coefficients of $B$, $G$, and $A$ are simple fractions and integers ($\pm 1, \pm \frac{1}{2}, 0$), the transformation steps require zero general multiplications—they are executed entirely with simple additions, subtractions, and bitwise arithmetic shifts.

The total number of multiplications drops from **$36$ down to $16$**—a **$2.25\times$ arithmetic speedup**.

## Minimal example
Computing a $3 \times 3$ convolution over a single channel producing an output feature map of size $28 \times 28$ ($14 \times 14 = 196$ output tiles of size $2 \times 2$):
- **Standard Direct Convolution:**
  $$196 \text{ tiles} \times 36 \text{ muls} = 7,056 \text{ multiplications}$$
- **Winograd $F(2 \times 2, 3 \times 3)$:**
  $$196 \text{ tiles} \times 16 \text{ muls} = 3,136 \text{ multiplications}$$
  The calculation achieves the exact same output values while eliminating nearly $4,000$ hardware multiplication cycles.

## Prerequisites
- [[concepts/macs-vs-flops|MACs vs FLOPs]]

## Common confusions
- **Winograd works efficiently for any kernel size:** Winograd's numerical stability and transformation overhead scale poorly as kernel size increases. It is exceptional for $3 \times 3$ and $5 \times 5$ filters with stride 1, but for larger kernels ($7 \times 7$) or strided convolutions, FFT-based convolutions or GEMM are preferred.
- **Filter transformation adds runtime latency:** Filter transformations ($G g G^T$) depend strictly on static weights and are pre-computed offline when compiling the model; only the input tile transform ($B^T d B$) and output transform ($A^T \dots A$) execute at runtime.

## Appears in
- [[lessons/2026-11-01-tinyengine-parallel-processing-and-inference-optimizations]]: introduced as an exact mathematical acceleration technique for convolutional inference.
