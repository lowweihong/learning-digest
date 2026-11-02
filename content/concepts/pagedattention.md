---
title: PagedAttention
aliases: [vLLM PagedAttention, KV cache paging, virtual memory KV cache, non-contiguous KV allocation]
tags: [concept, llm, systems, efficient-ai, serving]
---

## Definition
A memory management algorithm and attention kernel (the core technology behind **vLLM**) that stores intermediate [[concepts/kv-cache|KV Cache]] tokens in non-contiguous, fixed-size physical memory blocks managed via a logical page table, eliminating GPU memory fragmentation and enabling zero-copy cache sharing.

## Why it's true
In standard LLM serving runtimes, managing KV cache memory in GPU high-bandwidth memory (HBM) suffers from severe inefficiency:
1. **Unpredictable Output Lengths:** A user's prompt might produce a 5-word answer or a 1,000-word essay. Because the system cannot predict generation length in advance, it pre-allocates a contiguous block of memory sized to the maximum allowable sequence length (e.g., $2,048$ or $4,096$ tokens).
2. **Internal Fragmentation:** A request that terminates after 100 tokens leaves $95\%$ of its pre-allocated continuous buffer empty and locked against other users.
3. **External Fragmentation:** Over time, interleaved requests of varying lengths start and finish, creating fragmented memory holes that cannot accommodate large contiguous allocations.
In production deployments, these inefficiencies waste **$60\text{–}80\%$ of GPU memory**, drastically choking serving capacity and limiting batch concurrency.

**PagedAttention** adapts classical operating system virtual memory paging to transformer attention:
- **Fixed-Size Physical Blocks:** GPU memory is partitioned into uniform physical blocks, each holding the Keys and Values for a fixed number of tokens (typically 16 tokens per block).
- **Logical Page Tables:** Each incoming request maintains a lightweight logical page table that maps its sequential token positions to non-contiguous physical blocks scattered throughout VRAM.
- **On-Demand Allocation:** As new tokens are decoded, new physical blocks are allocated dynamically only when the current block is filled. Unused memory slots are restricted entirely to the very last block of a sequence, reducing memory waste to under $4\%$.
- **Zero-Copy Prefix Sharing:** When multiple generation requests share an identical prompt (e.g., parallel sampling or multi-turn chat), their logical tables point to the exact same physical prompt blocks. New physical memory is allocated only when tokens diverge (Copy-on-Write).

```
Logical Request A:  [ Block 0 ]  ===> Physical Pool: [ Block 4 (Tokens 0-15)  ]
                    [ Block 1 ]  ===>                 [ Block 9 (Tokens 16-31) ]

Logical Request B:  [ Block 0 ]  ===> (Shares Block 4! Zero Memory Duplication!)
                    [ Block 1 ]  ===>                 [ Block 2 (Tokens 16-31) ]
```

## Minimal example
Serving an interactive customer service assistant with a system prompt of $1,024$ tokens where 4 users ask simultaneous questions:
- **Traditional Contiguous Allocation:**
  Each request pre-allocates a $4,096$-token buffer:
  $$4 \text{ requests} \times 4096 \text{ tokens} \times 2.5\text{ MB/token} = 40.96\text{ GB VRAM}$$
- **PagedAttention Allocation:**
  The system prompt ($1,024$ tokens) is stored once in $64$ physical blocks ($2.56\text{ GB}$). All 4 users share those 64 blocks. As users generate an average of 100 new tokens, each allocates only $\lceil 100 / 16 \rceil = 7$ private blocks ($7 \times 4 \times 16 \times 2.5\text{ MB} = 1.12\text{ GB}$).
  Total memory consumed: $2.56\text{ GB} + 1.12\text{ GB} = 3.68\text{ GB}$—an **$11\times$ reduction in VRAM consumption**, allowing the same GPU to serve over $10\times$ more concurrent users.

## Prerequisites
- [[concepts/kv-cache|KV Cache]]

## Common confusions
- **PagedAttention changes attention scores:** PagedAttention calculates the mathematically exact self-attention distribution. It modifies only the physical memory lookup indices and pointer indirection inside the CUDA kernel, producing zero loss in generation quality.
- **Block size should be 1 token:** Setting block size to 1 eliminates all internal fragmentation, but causes severe GPU kernel launch overhead and low memory coalescing. A block size of 16 tokens balances near-zero memory waste ($<4\%$) with peak GPU memory-bus saturation.

## Appears in
- [[lessons/2026-11-01-llm-quantization-and-deployment-techniques]]: introduced as the memory virtualization architecture that revolutionized LLM inference throughput.
