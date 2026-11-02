---
title: KV Cache
aliases: [key-value cache, autoregressive generation cache, LLM memory footprint]
tags: [concept, llm, transformers, efficient-ai, systems]
---

## Definition
A runtime memory buffer that stores the intermediate Key ($K$) and Value ($V$) activation projections of past tokens across all transformer layers during autoregressive language model decoding, eliminating redundant recomputation and reducing per-token generation complexity from $O(N^2)$ down to $O(N)$.

## Why it's true
In decoder-only autoregressive language models (such as GPT-4 and LLaMA), generation proceeds token-by-token: each newly predicted token is appended to the context, and the model predicts the next token. Because causal attention prevents past tokens from looking into the future, the key and value representations of token $i$ depend strictly on tokens $1 \dots i$ and remain completely static across subsequent generation steps.

#### Generation Without KV Cache: $O(N^2)$ Per Token
If past activations are discarded after each step, generating token $t$ forces the model to feed the entire sequence of length $t-1$ through all layers from scratch to recompute keys and values. The arithmetic complexity for generating a sequence of length $N$ scales cubically:
$$\text{Total Compute} = \sum_{t=1}^N O(t^2) = O(N^3)$$
This causes token generation to slow to an unviable crawl as context length grows.

#### Generation With KV Cache: $O(N)$ Per Token
By retaining previously computed $K$ and $V$ tensors in GPU memory (the **KV Cache**):
1. **Incremental Projection:** At step $t$, the model only projects the single incoming token into query $q_t$, key $k_t$, and value $v_t$.
2. **Cache Concatenation:** Append $k_t$ and $v_t$ to the existing cache: $K_{1:t} = [K_{1:t-1}, k_t]$ and $V_{1:t} = [V_{1:t-1}, v_t]$.
3. **Linear Attention:** Compute attention for the single query $q_t$ against the cached keys and values:
   $$\text{Attention}(q_t, K_{1:t}, V_{1:t}) = \text{Softmax}\left(\frac{q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$
Per-token arithmetic complexity drops to $O(N)$, and total sequence generation drops to $O(N^2)$.

#### The Memory Bottleneck
The speedup comes at a massive memory cost. For a model with $L$ layers, $H$ heads, head dimension $D$, batch size $B$, and sequence length $N$ stored in 16-bit precision (2 bytes):
$$\text{KV Cache Size} = 2 \times B \times N \times L \times H \times D \times 2\text{ bytes}$$

On LLaMA-2 70B ($L=80, H=64, D=128$) with a modest batch size of 16 and a $4\text{K}$ context window:
$$\text{KV Cache Size} = 2 \times 16 \times 4096 \times 80 \times 64 \times 128 \times 2\text{ bytes} \approx 175.9\text{ GB}$$
The dynamic KV cache consumes far more VRAM than the static 140 GB model weights, making KV cache compression techniques like [[concepts/grouped-query-attention|Grouped-Query Attention]] mandatory for production deployment.

## Minimal example
Generating 500 tokens with LLaMA-2 7B ($L=32, H=32, D=128$, FP16):
- **Per-Token Cache Growth:** Each token in the context consumes:
  $$2 \times 32\text{ layers} \times 32\text{ heads} \times 128\text{ dim} \times 2\text{ bytes} = 524,288\text{ bytes} = 512\text{ KB per token}$$
- At sequence length $N = 2,048$ with batch size $1$, the cache consumes $2,048 \times 512\text{ KB} = 1.07\text{ GB}$.
- With batch size $32$, the cache consumes $34.3\text{ GB}$ of GPU VRAM, requiring an A100/H100 GPU simply to hold intermediate attention states.

## Prerequisites
- [[concepts/the-memory-compute-gap|The Memory-Compute Gap]]

## Common confusions
- **KV cache stores Query vectors:** Query vectors ($q_t$) are used immediately to attend over cached keys and are discarded after each step. Only Keys ($K$) and Values ($V$) are retained in the cache.
- **KV cache is needed during training:** During training, the entire ground-truth sequence is available at once. Training processes all tokens in parallel using causal masking, requiring zero KV caching. The KV cache is strictly an inference-time optimization.

## Appears in
- [[lessons/2026-11-01-transformers-llms-and-attention-mechanisms]]: introduced as the fundamental mechanism and memory bottleneck of autoregressive LLM decoding.
