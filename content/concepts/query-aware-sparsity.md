---
title: Query-Aware Sparsity
aliases: [QUEST, dynamic KV cache pruning, query-dependent sparse attention]
tags: [concept, llm, sparse-attention, quest, inference-acceleration]
---

## Definition
A dynamic inference mechanism that preserves the complete KV cache in memory, but selectively fetches and computes attention over only a small subset of the most critical token pages for each query token based on fast, pre-computed page-level key statistics.

## Why it's true
Many sparse attention approaches attempt to save resources by permanently evicting keys that exhibit low attention weights during earlier generation steps. However, static or history-based eviction suffers from a fatal flaw: **query-dependency**.

A token that appears completely irrelevant for hundreds of steps may instantly become the single most critical token in the sequence when the prompt asks a targeted question:
- *Context:* "Person A works in City B. Person C works in City D."
- *5,000 steps of irrelevant conversation pass.*
- *New Query:* "Where does Person A work?"

If Person A's location was evicted during the intervening conversation, the model cannot recover it.

**Query-Aware Sparsity (QUEST)** recognizes that decoding latency is governed by GPU High-Bandwidth Memory (HBM) bandwidth—the time spent streaming vast KV caches across the memory bus:
1. **Retain Full Cache:** Keep all key and value vectors in memory, ensuring no fact is ever permanently discarded.
2. **Compact Page Summaries:** Divide the cache into small pages (e.g., 16 tokens per page) and store lightweight summary vectors: the element-wise minimum ($\min_k$) and maximum ($\max_k$) of the keys in that page.
3. **Upper-Bound Importance Estimation:** When a new query token $q$ arrives during decoding, evaluate $q$ against the page summary vectors in fast local SRAM to establish an upper bound on the maximum attention score that page could achieve:
   $$\text{Score}_{\text{page}} = \sum_{d} \max(q_d \cdot \min_{k,d}, q_d \cdot \max_{k,d})$$
4. **Targeted Fetching:** Rank all pages according to this upper bound and load only the top $K$ pages (often just 1% to 5% of the total context) into the tensor cores for exact attention computation.

Because 95%+ of the KV cache is never transferred over the memory bus during each decoding step, latency drops dramatically without sacrificing retrieval accuracy.

## Minimal example
In a 128,000-token context partitioned into 8,000 pages of 16 tokens each:
- **Dense Decoding:** The GPU must read all 8,000 pages (tens of gigabytes) from memory for **every single generated token**, choking memory bandwidth.
- **QUEST Decoding:** The query vector evaluates 8,000 tiny summary pairs in SRAM, selects the top 80 pages with the highest bound, and transfers only those 80 pages (1% of the cache) from HBM.
- **Retrieval Test:** On Needle In A Haystack benchmarks, QUEST matches the 100% accuracy of full dense attention while running up to $7\times$ faster.

## Prerequisites
- [[concepts/kv-cache|KV Cache]]: The key-value store that QUEST selectively queries to circumvent memory bandwidth limits.

## Common confusions
- **QUEST reduces GPU memory usage:** QUEST does **not** reduce memory allocation; it retains 100% of the KV cache in RAM to avoid catastrophic forgetting. It optimizes **memory bandwidth and decoding latency** by avoiding unnecessary memory transfers.
- **QUEST approximates attention values:** QUEST only approximates the *page selection ranking*. Once the top pages are chosen, it computes full, exact, mathematically uncompromised softmax attention over the tokens within those pages.

## Appears in
- [[lessons/2026-11-04-long-context-llms-context-extension-and-sparse-attention]]: detailed as an exact-accuracy decoding accelerator for memory-bandwidth-limited long-context inference.
