---
title: Grouped-Query Attention
aliases: [GQA, grouped query attention, multi-query attention, MQA vs GQA]
tags: [concept, llm, transformers, attention, efficient-ai]
---

## Definition
An attention architecture variant that partitions query heads into distinct groups where all query heads in a group share a single Key and Value head, striking an optimal balance between the high quality of **Multi-Head Attention (MHA)** and the minimal memory footprint of **Multi-Query Attention (MQA)**.

## Why it's true
In large language model generation, the dynamic [[concepts/kv-cache|KV Cache]] is often the dominant consumer of GPU high-bandwidth memory (HBM). Comparing attention topologies reveals the trade-offs:

1. **Multi-Head Attention (MHA):** Every query head has its own private key and value head ($H$ queries, $H$ keys, $H$ values). While expressive, it produces the largest possible KV cache ($H$ key-value tensors per layer per token). Memory bandwidth bottlenecks during memory-bound decoding severely constrain generation throughput.
2. **Multi-Query Attention (MQA):** All $H$ query heads share a single, global key head and value head ($H$ queries, $1$ key, $1$ value). This slashes KV cache size and memory traffic by $H\times$ (e.g., $32\times$ reduction). However, collapsing all context into a single key-value projection causes severe accuracy drops on nuanced downstream reasoning tasks.
3. **Grouped-Query Attention (GQA):** Divides the $H$ query heads into $G$ intermediate groups (where $1 < G < H$). Each group shares one key head and one value head:
   $$\text{Queries per KV Head} = \frac{H}{G}$$

```
Multi-Head Attention (MHA):     Grouped-Query Attention (GQA):      Multi-Query Attention (MQA):
Q Q Q Q Q Q Q Q                 [Q Q] [Q Q] [Q Q] [Q Q]             Q Q Q Q Q Q Q Q
| | | | | | | |                   \ /   \ /   \ /   \ /              \ \ \ | / / / /
K K K K K K K K                    K     K     K     K                      K
| | | | | | | |                    |     |     |     |                      |
V V V V V V V V                    V     V     V     V                      V
(Full Memory, High Accuracy)    (8x Smaller Cache, Full Accuracy)   (Minimal Memory, Quality Drops)
```

By setting $G = 8$ on a 64-head model (as in LLaMA-2 70B and LLaMA-3), GQA slashes the KV cache memory footprint and memory-loading traffic by **$8\times$**, while empirical evaluations show zero degradation in perplexity or task performance compared to full MHA.

## Minimal example
Evaluating inference on a 70B foundation model with 64 attention heads ($H=64, D=128$) running FP16 at sequence length $N = 4,096$ with batch size $16$:
- **MHA ($64$ KV heads):**
  $$\text{KV Cache} = 2 \times 16 \times 4096 \times 80\text{ layers} \times 64\text{ heads} \times 128\text{ dim} \times 2\text{ bytes} \approx 175.9\text{ GB}$$
  Requires at least three 80 GB A100 GPUs solely to store the KV cache.
- **GQA ($G = 8$ KV heads, 8 queries per group):**
  $$\text{KV Cache} = 2 \times 16 \times 4096 \times 80\text{ layers} \times 8\text{ heads} \times 128\text{ dim} \times 2\text{ bytes} \approx 22.0\text{ GB}$$
  The KV cache shrinks by an exact factor of $8\times$, fitting comfortably alongside model weights on a single multi-GPU node and dramatically accelerating decoding latency.

## Prerequisites
- [[concepts/kv-cache|KV Cache]]

## Common confusions
- **GQA reduces the number of Query heads:** GQA preserves the exact same number of query heads $H$ as MHA, maintaining full multi-head expressive power on the query side. Only Key and Value heads are grouped and shared.
- **Can pre-trained MHA models be converted to GQA?** Yes. Researchers often convert pre-trained MHA checkpoints into GQA models by mean-pooling the weights of adjacent key/value heads and performing brief post-training (uptraining), avoiding the cost of pre-training from scratch.

## Appears in
- [[lessons/2026-11-01-transformers-llms-and-attention-mechanisms]]: introduced as the modern standard architecture for reducing LLM KV cache memory traffic.
