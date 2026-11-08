---
title: Lost in the Middle
aliases: [lost in the middle phenomenon, U-shaped retrieval curve, context depth degradation]
tags: [concept, llm, evaluation, benchmarks, long-context]
---

## Definition
A systematic failure pattern in long-context language models where information retrieval and reasoning performance follows a U-shaped curve: accuracy is highest when target evidence appears near the very beginning or the very end of the prompt, but drops sharply when the evidence is buried in the middle.

## Why it's true
A language model that accepts 128,000 tokens without crashing does not necessarily attend to or utilize all 128,000 tokens equally. The "Lost in the Middle" phenomenon arises from two compounding biases inherent in decoder-only Transformer architectures:

1. **Primacy Bias (The Start):** The initial tokens of a prompt define system instructions, role descriptions, and the formatting contract. Due to causal attention masking, these early tokens are attended to by every subsequent token in every layer of the network. As a result, early context benefits from disproportionately strong and consistent gradient signals throughout pre-training.
2. **Recency Bias (The End):** Tokens placed right before the query have the smallest relative positional distance (under RoPE or relative attention). Their key-query dot products do not suffer from positional dispersion, and their representations have not experienced repeated dilution through multiple causal aggregation steps.
3. **The Middle Valley (Depths 20% to 80%):** Information positioned in the middle of long documents enjoys neither the universal visibility of the beginning nor the positional sharpness of the end. When a query computes attention across tens of thousands of tokens, the softmax distribution over middle keys flattens into numerical noise. 

Consequently, while the model generates perfectly grammatical and plausible-sounding sentences, it routinely fails to retrieve factual needles hidden in the middle of long haystacks.

```mermaid
xychart-beta
    title "Retrieval Accuracy vs. Evidence Depth in Long Context"
    x-axis "Evidence Position in Context" [0%, 20%, 40%, 60%, 80%, 100%]
    y-axis "Accuracy (%)" 0 --> 100
    line [98, 48, 22, 25, 52, 95]
```

## Minimal example
In a controlled test, a 32,000-token prompt is composed of unrelated academic papers, with a single target sentence inserted: *"The access code for Vault 9 is 7492."*
- **Target at 5% Depth (Start):** The model answers correctly: `7492` (99% success rate across trials).
- **Target at 95% Depth (End):** The model answers correctly: `7492` (97% success rate).
- **Target at 50% Depth (Middle):** The model hallucinates: *"The document does not provide the access code"* or guesses an incorrect number (success rate drops below 30%).

## Prerequisites
- None

## Common confusions
- **Large context window implies deep comprehension:** Extending a model's context window from 8K to 128K via RoPE interpolation allows the KV cache to store more tokens without out-of-memory crashes, but does not solve the U-shaped retrieval deficit without targeted multi-position training and synthetic needle-retrieval fine-tuning.
- **Lost in the middle is solved by higher parameter counts:** Even 70B+ parameter foundation models exhibit the U-shaped curve when tested across context windows exceeding their native pre-training sweet spot.

## Appears in
- [[lessons/2026-11-04-long-context-llms-context-extension-and-sparse-attention]]: examined as the primary motivation for rigorous evaluation protocols like Needle In A Haystack (NIAH) and LongBench.
