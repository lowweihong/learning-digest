---
title: Shifted Sparse Attention
aliases: [S2-Attn, shifted short attention, longlora sparse attention]
tags: [concept, llm, fine-tuning, sparse-attention, longlora]
---

## Definition
A block-sparse attention pattern for efficient long-context fine-tuning that partitions attention heads into two equal groups: the first group attends within regular, non-overlapping local chunks, while the second group cyclically shifts the token sequence by half a chunk size before grouping, enabling cross-block information to propagate across layers without quadratic attention costs.

## Why it's true
Extending a pre-trained language model from a short context window (e.g., 4,096 tokens) to a long context window (e.g., 32,768 tokens) requires fine-tuning. However, full self-attention scales quadratically with sequence length:
$$\text{Attention FLOPs} = O(N^2 \cdot d)$$

Quadrupling the sequence length increases attention compute by $16\times$, turning fine-tuning into an operation requiring hundreds of enterprise GPUs.

Naive local chunking solves the computation issue by restricting attention to independent blocks of size $G$:
$$\text{Chunked FLOPs} = O(N \cdot G \cdot d)$$
However, independent chunking isolates groups of tokens into impenetrable silos: a token in chunk 1 can never pass information to a token in chunk 2.

**Shifted Sparse Attention ($S^2$-Attn)**, introduced in LongLoRA, breaks this isolation without increasing FLOPs:
1. **Split Heads:** Divide the attention heads into two equal sets ($H/2$ each).
2. **Regular Partitioning:** Set 1 attends strictly within fixed local chunks of size $G$:
   $$\text{Blocks}_1 = [0, G), [G, 2G), [2G, 3G), \dots$$
3. **Shifted Partitioning:** Set 2 shifts token positions by $G/2$ before applying the same local grouping:
   $$\text{Blocks}_2 = [G/2, 3G/2), [3G/2, 5G/2), \dots$$
4. **Layer-by-Layer Diffusion:** Because the blocks overlap by exactly half a chunk, tokens near the boundary of one group sit in the interior of the other. In Layer 1, tokens within each neighborhood share information. By Layer 2 and Layer 3, information hops across overlapping neighborhoods, establishing global information diffusion across the entire sequence.

During inference, the model seamlessly transitions back to full dense attention, preserving exact retrieval quality.

## Minimal example
Consider fine-tuning a 4,000-token sequence with chunk size $G = 1,000$:
- **Group 1 Heads:** Tokens $[0, 999]$ attend only to each other; tokens $[1000, 1999]$ attend only to each other.
- **Group 2 Heads:** The sequence is shifted by 500, so tokens $[500, 1499]$ attend together.
- Token 600 attends to Token 200 in Group 1, and to Token 1,200 in Group 2. In a single layer, Token 600 acts as an informational bridge between the first and second chunks, cutting attention compute by $4\times$ ($O(N \cdot 1000)$ instead of $O(N \cdot 4000)$).

## Prerequisites
- [[concepts/low-rank-adaptation|Low-Rank Adaptation]]: LongLoRA pairs shifted sparse attention with LoRA weight updates on projection layers to fine-tune 32K context windows on a single 8-GPU node.

## Common confusions
- **$S^2$-Attn is used during inference:** $S^2$-Attn is exclusively a **training and fine-tuning accelerator**. At test time, the model executes full dense attention to maintain maximum precision across long documents.
- **LoRA alone is sufficient for long context extension:** Standard LoRA saves parameter gradient memory, but the forward and backward passes of self-attention still evaluate the full $N \times N$ matrix. Without sparse attention like $S^2$-Attn, activation memory blows out GPU capacity.

## Appears in
- [[lessons/2026-11-04-long-context-llms-context-extension-and-sparse-attention]]: introduced as the core architectural component of LongLoRA for accessible context extension.
