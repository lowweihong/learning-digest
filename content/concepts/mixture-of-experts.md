---
title: Mixture of Experts
aliases: [MoE, sparse mixture of experts, conditional computation, sparse routing]
tags: [concept, llm, transformers, efficient-ai, scaling]
---

## Definition
A sparse neural network architecture and conditional computation framework that replaces dense feed-forward layers with multiple parallel sub-networks (**experts**), dynamically routing each individual token through only a top-$k$ subset of experts using a learned gating router.

## Why it's true
In standard dense neural networks, every parameter in every layer is actively evaluated for every single token processed. As models scale from 7B to 70B to 500B parameters, inference latency, memory bandwidth, and FLOP costs scale linearly, making massive dense models prohibitively slow and expensive to serve.

**Mixture of Experts (MoE)** decouples total model capacity (knowledge storage) from per-token inference cost:
1. **Parallel Expert Sub-Networks:** Replace the single feed-forward network (FFN) in each transformer block with $E$ independent expert FFNs (e.g., $E = 8$ or $E = 64$).
2. **Dynamic Top-$k$ Router:** For each token representation $x$, a lightweight linear routing layer computes affinity scores across all $E$ experts:
   $$G(x) = \text{Softmax}(\text{TopK}(x \cdot W_r, k))$$
3. **Sparse Execution:** The token is dispatched strictly to the top-$k$ selected experts (typically $k=1$ or $k=2$). The remaining $E - k$ experts are completely skipped for that token:
   $$\text{Output} = \sum_{i \in \text{TopK}} G(x)_i \cdot \text{Expert}_i(x)$$
4. **Asymmetric Scaling:** An MoE model with 8 experts each containing 7B parameters possesses over $47\text{B}$ total parameters in memory, but executes only $13\text{B}$ active parameters per token. The model achieves the reasoning capability of a 70B model while executing at the inference latency of a 13B model.

#### The Load Balancing & Capacity Factor Challenge
If some experts are favored early during training, a positive feedback loop can emerge where the router sends all tokens to the same two "popular" experts, leaving the others un-utilized. Systems enforce two guards:
- **Auxiliary Load-Balancing Loss:** Penalizes uneven routing distributions during training.
- **Expert Capacity Factor:** Sets a maximum threshold on how many tokens a single expert can accept per batch (e.g., $1.25 \times \text{average}$). If an expert reaches capacity, overflow tokens bypass the expert via residual connections.

## Minimal example
Comparing Mixtral 8x7B to a dense 70B model:
- **Total Parameters:** Mixtral has $46.7\text{B}$ total parameters (8 experts per layer).
- **Active Parameters per Token ($k=2$):** Only $12.9\text{B}$ parameters are activated during each forward pass.
- **Inference Speed:** Mixtral processes generation at approximately $3\times$ higher token throughput than dense LLaMA-2 70B while matching or exceeding its accuracy on MMLU, GSM8K, and HumanEval benchmarks.

## Prerequisites
- None

## Common confusions
- **MoE saves GPU VRAM:** While MoE slashes computational FLOPs and execution latency, **it does not reduce total memory consumption**. All expert weights must reside in GPU memory (or offloaded via high-speed NVLink/PCIe), meaning a 47B MoE still demands enough VRAM to store 47B weights.
- **Experts specialize neatly by human domains:** While it is tempting to assume Expert 1 learns French and Expert 2 learns Python, empirical inspections reveal that experts specialize in subtle syntactic patterns, token punctuation, and semantic abstraction levels rather than clean human subjects.

## Appears in
- [[lessons/2026-11-01-transformers-llms-and-attention-mechanisms]]: introduced as the leading architectural paradigm for scaling foundation models efficiently.
