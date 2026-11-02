---
title: Hybrid Attention NAS
aliases: [Jet-Nemotron, LLM architecture search, hybrid attention search]
tags: [concept, llm, nas, efficient-ai, transformers]
---

## Definition
The application of neural architecture search to large language models to optimize the layer-wise placement of full quadratic self-attention, sliding-window attention, and linear state-space models (SSMs), maximizing generation throughput and bounding KV-cache memory while preserving reasoning accuracy.

## Why it's true
In standard transformer LLMs, full self-attention scales quadratically with context length ($O(N^2)$), causing Key-Value (KV) cache memory to explode into tens of gigabytes during long-context generation. Pure linear attention architectures (such as Mamba or RWKV) achieve $O(1)$ constant memory and ultra-fast throughput, but suffer steep accuracy drops on multi-hop associative recall and complex reasoning benchmarks.

Hybrid Attention NAS (such as Song Han's **Jet-Nemotron** and NVIDIA's **FlexiTron**) solves this trade-off:
- It creates a supernet where each transformer layer can select between full attention, sliding-window attention, or linear gated delta networks.
- An evolutionary search identifies critical anchor layers where full attention is indispensable, while converting the remaining $80\%\text{–}90\%$ of layers into high-speed linear attention blocks.

## Minimal example
On an H100 GPU running long-context generation ($64\text{k}$ tokens):
- **Full Attention baseline (Llama-3 / Qwen-2.5):** Requires $1,800\text{ MB}$ of KV cache per sequence and runs at 240 tokens/sec.
- **Pure Linear Attention (Mamba-2 / RWKV):** Runs at 2,800 tokens/sec, but MMLU accuracy collapses from $60.2\%$ down to $41.3\%$.
- **Jet-Nemotron (Hybrid NAS):** Discovers that retaining full attention in only 2 or 3 strategic layers restores MMLU accuracy to $61.1\%$ while shrinking KV-cache memory to $200\text{ MB}$ and boosting throughput to **2,800 tokens/sec** ($47\times$ speedup over full attention).

## Prerequisites
- [[concepts/once-for-all-network|Once-for-All Network]]

## Common confusions
- **Uniform vs. NAS-placed hybrid layers:** Hand-crafting a uniform pattern (e.g., placing one full attention layer every 4 layers) achieves sub-optimal accuracy. NAS reveals that attention sensitivity is heavily skewed toward specific early and late boundary layers, demanding asymmetric placement.

## Appears in
- [[lessons/2026-11-01-hardware-aware-nas-and-once-for-all-networks]]: introduced as the modern application of NAS to large language model inference acceleration.
