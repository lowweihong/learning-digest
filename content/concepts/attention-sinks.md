---
title: Attention Sinks
aliases: [attention sink phenomenon, streaming attention sinks, initial token sink]
tags: [concept, llm, attention, memory, streaming]
---

## Definition
Initial tokens in an autoregressive sequence that receive massive, disproportionately high attention probability scores regardless of their semantic content, acting as an overflow valve for excess softmax probability when a query has no strong reason to attend to any past token.

## Why it's true
In standard self-attention, the softmax operator enforces an absolute mathematical constraint: for every query token $q_i$, the attention weights over all accessible keys $k_j$ must sum precisely to $1$:
$$\sum_{j=1}^{i} \text{Softmax}\left(\frac{q_i k_j^T}{\sqrt{d}}\right) = 1$$

During language generation, many query tokens (such as punctuation, common conjunctions, or filler transitions) do not require semantic information from earlier context. However, because softmax exponentiates dot products ($e^{z} > 0$), a query cannot assign an attention weight of zero across the board. The model *must* dump the excess probability somewhere.

Because of causal attention masking, earlier positions are visible to **every single subsequent token** throughout the entire document:
1. **Universal Visibility:** Token 0 is present in the receptive field of every layer and every head from beginning to end.
2. **Learned Dumping Ground:** Optimization naturally leads the model to designate these initial tokens as universal "attention sinks," absorbing upwards of 30% to 80% of total attention mass even when they contain nothing more than a starting prompt delimiter (`<s>`) or a newline.
3. **The Catastrophic Window Eviction Trap:** When running long generation with a naive sliding-window KV cache (evicting older tokens to fit inside GPU memory), dropping token 0 removes this learned overflow valve. The softmax denominator suddenly redistributes this massive excess probability across the remaining local tokens in ways never seen during pre-training, causing perplexity to explode and output to degenerate into gibberish.

Retaining just 4 initial sink tokens alongside the local rolling window stabilizes the attention distribution indefinitely, unlocking infinite streaming with $O(1)$ constant memory.

## Minimal example
Imagine generating text for 50,000 steps using a sliding context window of 1,024 tokens:
- **Naive Sliding Window (tokens 49,000 to 50,024 retained):** Because token 0 was evicted at step 1,025, the model's perplexity jumps from 5.4 to over 1,000 within a few dozen steps, producing repetitive garbage.
- **StreamingLM (tokens 0, 1, 2, 3 + rolling tokens 49,004 to 50,024 retained):** By preserving the 4 initial sink tokens in the KV cache, the model maintains a perplexity of 5.4 throughout all 50,000 steps without increasing memory consumption by a single byte.
- **Content Irrelevance Test:** Replacing the initial 4 tokens with four meaningless newline characters (`\n\n\n\n`) yields the exact same stable perplexity, proving that the model relies on the positional sink rather than the semantic text.

## Prerequisites
- [[concepts/kv-cache|KV Cache]]: The memory store holding past key and value vectors that attention sinks stabilize during rolling generation.

## Common confusions
- **Attention sinks represent the most important words:** Sinks do not indicate high semantic importance. They represent mathematical artifacts of softmax normalization acting as probability sinks for queries that don't need context.
- **Attention sinks are only found in LLMs:** Sinks also appear in Vision Transformers (ViTs) and Diffusion Transformers, where patches representing blank background space (such as empty sky or solid walls) act as spatial attention sinks absorbing excess attention weight.

## Appears in
- [[lessons/2026-11-04-long-context-llms-context-extension-and-sparse-attention]]: analyzed as the core mechanism enabling StreamingLM and infinite streaming inference under bounded memory budgets.
