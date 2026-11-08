---
title: Dual Attention
aliases: [dual-attention architecture, retrieval and streaming heads, split-head sparse attention]
tags: [concept, llm, sparse-attention, inference-acceleration, kv-cache]
---

## Definition
An attention architecture that splits transformer attention heads into two distinct functional categories: dense **retrieval heads** that maintain a complete KV cache for exact long-distance recall, and **streaming heads** that maintain only attention sinks and a local sliding window, cutting overall KV memory consumption by over 50% without degrading long-context accuracy.

## Why it's true
In standard multi-head attention, every head stores key and value vectors for every single generated token. For million-token contexts, this uniform storage creates an unsustainable memory bottleneck:
$$\text{KV Cache Memory} = 2 \times B \times L \times H_{KV} \times d_{head} \times N$$

However, different attention heads do not perform the same job:
1. **Functional Division of Labor:** Probing attention heads reveals that only a small minority (~20%–40%) actively perform long-range factual retrieval (matching queries to distant needles). The majority of heads focus strictly on local syntactic dependencies, next-token smoothing, and phrase agreement.
2. **Head Specialization via Distillation:** Dual Attention learns which heads require global history by training a gating scalar $\alpha_h \in [0, 1]$ for each head via attention distillation on long documents:
   $$\text{Output}_h = \alpha_h \cdot \text{FullAttn}(Q, K, V) + (1 - \alpha_h) \cdot \text{StreamingAttn}(Q, K, V)$$
   With an L1 regularization penalty driving $\alpha_h$ toward 0, heads that do not strictly require full history naturally drop to low $\alpha$.
3. **Decoupled Cache Layouts:** At inference time:
   - **Retrieval Heads ($\alpha_h \ge \tau$):** Maintain full dense KV cache across all $N$ tokens to guarantee 100% retrieval on needle-in-a-haystack tasks.
   - **Streaming Heads ($\alpha_h < \tau$):** Retain only $S$ initial [[concepts/attention-sinks|Attention Sinks]] (e.g., 4 tokens) and a local window $W$ (e.g., 512 tokens), permanently evicting all intermediate tokens.

Because the majority of heads transition to fixed-size streaming buffers, KV cache memory and decoding bandwidth scale far more gently with context length.

## Minimal example
Consider an LLM with 32 attention heads running on a 1,000,000-token sequence:
- **Standard Dense Attention:** All 32 heads retain 1,000,000 tokens of keys and values, generating $32 \times 10^6$ stored token vectors (~100 GB in FP16, exceeding an 80 GB GPU).
- **Dual Attention:**
  - 10 heads are assigned as **Retrieval Heads**: $10 \times 1,000,000 = 10,000,000$ stored tokens.
  - 22 heads are assigned as **Streaming Heads**: $22 \times (4 \text{ sinks} + 512 \text{ window}) \approx 11,352$ stored tokens.
  - Total stored vectors: $\approx 10.01 \times 10^6$ tokens.

Memory usage drops by **68%**, fitting a 1-million-token context inside a single A100/H100 GPU while preserving full factual retrieval accuracy.

## Prerequisites
- [[concepts/kv-cache|KV Cache]]: The memory store holding key and value states that Dual Attention partially evicts.
- [[concepts/attention-sinks|Attention Sinks]]: The foundational stability mechanism that allows streaming heads to drop older tokens without numerical collapse.

## Common confusions
- **Dual Attention drops distant facts if they fall in streaming heads:** Retrieval heads maintain the complete sequence history. Because key routing assigns all long-distance associative recall to retrieval heads, factual needles anywhere in the 1M sequence are recalled with 100% fidelity.
- **Dual Attention modifies model weights:** Dual Attention does not modify feedforward weights or projection matrices. The gating distillation process tunes only ~1,000 scalar gate parameters (one per head per layer), taking only a few GPU hours.

## Appears in
- [[lessons/2026-11-04-long-context-llms-context-extension-and-sparse-attention]]: introduced as a state-of-the-art serving technique for scaling language model contexts into millions of tokens.
