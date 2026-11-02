---
title: Loop Tiling and Reordering
aliases: [loop blocking, loop interchange, cache locality optimization, IKJ matrix multiplication]
tags: [concept, systems, parallel-computing, computer-architecture, efficient-ai]
---

## Definition
Compiler- and hardware-level loop transformation techniques that dramatically reduce memory latency by reordering nested loop indices to access memory sequentially (**loop reordering / interchange**) and partitioning iteration spaces into sub-blocks that fit entirely within high-speed CPU/GPU caches (**loop tiling / blocking**).

## Why it's true
Modern processors suffer from the memory-compute gap: performing an arithmetic multiply-accumulate takes 1 clock cycle, but fetching data from external DRAM takes hundreds of cycles. High performance relies on keeping data in fast on-chip caches (L1, L2, L3).

In naive matrix multiplication ($C = A \times B$):
```c
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < N; k++)
      C[i][j] += A[i][k] * B[k][j];
```
1. **The Row-Major Stride Penalty:** In C/C++, matrices are stored row-by-row in continuous memory. The inner loop increments $k$, accessing $B[k][j]$. This jumps an entire row ($N$ elements) on every single clock cycle. Each access misses the CPU cache, wasting memory bandwidth.
2. **Loop Reordering ($I \to K \to J$):** Switching the inner two loops so $j$ is the innermost variable makes $B[k][j]$ a sequential stride-1 access. A single 64-byte cache line fetch loads multiple adjacent numbers that are consumed in consecutive cycles, yielding an immediate $12\times$ speedup without extra memory.
3. **Loop Tiling (Blocking):** When matrices are larger than cache capacity (e.g., $1024 \times 1024$ floats $= 4\text{ MB}$, far exceeding a $32\text{ KB}$ L1 cache), elements loaded into cache are evicted before they can be reused by subsequent outer loop iterations. Tiling breaks the loops into sub-matrices of size $B_{\text{tile}} \times B_{\text{tile}}$ designed to fit inside L1 cache. The processor reuses loaded tiles completely before moving to the next block, achieving up to $20\times$ additional acceleration.

## Minimal example
Benchmarking a $1024 \times 1024$ float32 matrix multiplication on a modern CPU:
- **Naive ($I \to J \to K$):** High cache miss rates on matrix $B$; execution time $= 24.1\text{ s}$.
- **Reordered ($I \to K \to J$):** Sequential cache line access on $A$ and $B$; execution time drops to $1.9\text{ s}$ ($12.6\times$ speedup).
- **Tiled ($I, J, K$ tiled with $32 \times 32$ blocks for L1 cache):** Maximized temporal data reuse; execution time drops to $0.11\text{ s}$ ($219\times$ cumulative speedup over naive code).

## Prerequisites
- [[concepts/the-memory-compute-gap|The Memory-Compute Gap]]

## Common confusions
- **Tiling changes mathematical results:** Tiling and reordering change only the scheduling order of associative additions; the mathematical result of the matrix multiplication remains identical.
- **Tiling size should be as large as possible:** If the tile size exceeds L1 cache capacity, cache lines are constantly evicted and reloaded ("cache thrashing"), destroying performance. Tile sizes must be precisely tuned to hardware cache limits.

## Appears in
- [[lessons/2026-11-01-tinyengine-parallel-processing-and-inference-optimizations]]: introduced as the fundamental software optimization for linear algebra and deep learning kernels.
