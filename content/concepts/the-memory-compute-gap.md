---
title: The Memory-Compute Gap
aliases: [memory wall, compute-memory bottleneck, memory gap]
tags: [concept, computer-architecture, efficiency]
---

## Definition
The growing divergence between how fast AI models are expanding in parameter size and how slowly hardware memory capacity and memory bandwidth are increasing.

## Why it's true
Over the past decade, AI models grew by thousands of times (from AlexNet's 60 million parameters to trillion-parameter LLMs). But GPU memory only grew from 16 GB (NVIDIA P100) to around 192 GB (B200)—a steady but far slower climb. Because fetching a number from memory costs up to 200× more energy than multiplying it, modern AI is bottlenecked by the time and power it takes to shuffle numbers across memory wires, not by raw math.

## Minimal example
Running a 70-billion parameter model at 16-bit precision requires 140 GB just to hold the weights. A high-end consumer GPU (like an RTX 4090) only has 24 GB of memory. Even if the chip's computing cores are blazing fast, the model cannot even load into memory without compression techniques like quantization or pruning.

## Prerequisites
- None

## Common confusions
- Assuming AI models are slow because the math is complicated: The math is mostly simple multiplication and addition. The slowdown and battery drain come from the memory bus struggling to deliver billions of numbers to the processor on time.

## Appears in
- [[lessons/2026-11-01-intro-to-tinyml-and-efficient-ai]]: motivates why model compression and efficiency techniques are essential.
