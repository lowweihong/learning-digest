---
title: SIMD Vectorization
aliases: [SIMD, single instruction multiple data, vector instructions, AVX, NEON, packed arithmetic]
tags: [concept, systems, hardware, parallel-computing, computer-architecture]
---

## Definition
A hardware architecture and instruction set design where a single CPU instruction performs the same mathematical operation across multiple data elements simultaneously using wide vector registers (e.g., 128-bit SSE/NEON, 256-bit AVX2, or 512-bit AVX-512).

## Why it's true
In classical scalar computing (Single Instruction, Single Data or SISD), executing a multiply-accumulate on four 32-bit floats requires four independent instructions:
```
Scalar:  mul r1, r2 -> r3;  mul r4, r5 -> r6;  mul r7, r8 -> r9;  mul r10, r11 -> r12;
```
Each instruction incurs instruction-fetch overhead, decode overhead, and separate pipeline dispatch cycles.

SIMD exploits data-level parallelism:
1. **Packed Vector Registers:** A 128-bit register packs four 32-bit float values (or sixteen 8-bit integers) side-by-side.
2. **Single-Cycle Parallel ALUs:** A single SIMD instruction (e.g., `_mm_mul_ps` on x86 or `vmulq_f32` on ARM NEON) dispatches all four multiplications simultaneously into four hardware execution lanes.
3. **Throughput Scaling:** Arithmetic throughput scales linearly with the vector width: $4\times$ speedup for 128-bit registers, $8\times$ for 256-bit registers (AVX2), and up to $16\times$ for 8-bit quantized integers in microcontroller DSP extensions (like ARM Helium / Cortex-M SIMD).
4. **Memory Alignment Necessity:** SIMD operates at peak efficiency when data is stored contiguously in memory and aligned to cache boundaries, enabling packed vector loads (`_mm_load_ps`) rather than slow gather/scatter operations.

## Minimal example
Multiplying two float arrays of length 1,024:
- **Scalar Loop:** Requires 1,024 scalar load instructions and 1,024 multiply instructions across 1,024 loop iterations.
- **128-bit ARM NEON SIMD Loop:**
  ```c
  for (int i = 0; i < 1024; i += 4) {
    float32x4_t va = vld1q_f32(&A[i]);
    float32x4_t vb = vld1q_f32(&B[i]);
    float32x4_t vc = vmulq_f32(va, vb);
    vst1q_f32(&C[i], vc);
  }
  ```
  The loop executes in only 256 iterations, issuing $4\times$ fewer instructions and delivering near $4\times$ higher arithmetic throughput.

## Prerequisites
- [[concepts/the-memory-compute-gap|The Memory-Compute Gap]]

## Common confusions
- **SIMD vs. Multi-Threading:** Multi-threading distributes different instructions and tasks across multiple CPU cores. SIMD executes a single instruction across multiple data lanes *inside a single core*. They are orthogonal and combined in high-performance engines (e.g., OpenMP across cores + SIMD inside each core).
- **Compilers always vectorize automatically:** While compilers attempt auto-vectorization, complex pointer aliasing, non-stride-1 memory access, or loop-carried dependencies frequently prevent automatic SIMD generation, requiring explicit intrinsics or restructuring.

## Appears in
- [[lessons/2026-11-01-tinyengine-parallel-processing-and-inference-optimizations]]: introduced as the hardware instruction acceleration mechanism used in CPU and MCU neural network runtimes.
