---
title: Latency vs Throughput
aliases: [latency, throughput, batching trade-off]
tags: [concept, computer-systems, performance]
---

## Definition
**Latency** is the time required to complete a single task from start to finish (measured in milliseconds), while **Throughput** is the total number of tasks completed per unit of time (measured in queries, images, or tokens per second).

## Why it's true
A system can achieve high throughput while having worse (longer) latency. When you process requests in batches using parallel processor cores, each individual request may sit in a queue or take longer to finish, but the hardware produces more total outputs per second. In interactive systems (like self-driving cars or chatbots), users care about low latency; in offline systems (like indexing photos overnight), developers care about high throughput.

## Minimal example
Imagine two photo processing systems:
- **System A (Sequential):** Has 1 core. Each photo takes 50 ms.
  - Latency = **50 ms**.
  - Throughput = $1,000 \text{ ms} / 50 \text{ ms} = \mathbf{20 \text{ images/sec}}$.
- **System B (Parallel):** Has 4 cores running in parallel. Each photo takes 100 ms.
  - Latency = **100 ms** (2× slower!).
  - Throughput = $4 \times (1,000 \text{ ms} / 100 \text{ ms}) = \mathbf{40 \text{ images/sec}}$ (2× higher throughput!).

## Prerequisites
- None

## Common confusions
- Assuming faster throughput automatically means lower latency: Throughput often increases by batching multiple inputs together, which increases queue wait times and worsens individual latency.

## Appears in
- [[lessons/2026-11-01-basics-of-neural-networks-and-efficiency-metrics]]: contrasts real-time responsiveness with bulk processing capacity.
