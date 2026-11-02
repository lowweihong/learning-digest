---
title: Speculative Decoding
aliases: [speculative sampling, draft and verify, draft model decoding, speculative execution LLM]
tags: [concept, llm, inference, efficient-ai, systems]
---

## Definition
An inference acceleration paradigm that pairs a large, high-quality **target model** with a compact, ultra-fast **draft model**, using the draft model to propose candidate token sequences cheaply and the target model to verify them all in a single parallel forward pass, delivering $2\times$ to $3\times$ speedups with mathematically lossless output quality.

## Why it's true
In interactive LLM decoding (batch size 1), generation is strictly **memory-bandwidth bound**: reading the tens of billions of weights from GPU high-bandwidth memory (HBM) takes orders of magnitude longer than executing the arithmetic. A 70B model must stream 140 GB of data through its memory bus to emit a single token.

However, once a sequence of tokens is known in advance, modern GPUs evaluate multiple tokens in parallel at virtually zero additional memory cost (identical to prompt prefill), because the model weights only need to be read from HBM once.

**Speculative Decoding** leverages this asymmetry through a two-phase loop:
1. **Draft Phase:** A small draft model (e.g., a 1B model, $50\times$ smaller and faster) autoregressively proposes $K$ candidate tokens: $(\hat{x}_1, \hat{x}_2, \dots, \hat{x}_K)$.
2. **Verification Phase:** The large target model processes all $K$ draft tokens concurrently in a **single forward pass** using causal masking, generating true probability distributions $P(x_i)$ for all $K$ positions simultaneously.
3. **Rejection Sampling:** The system evaluates candidate tokens sequentially from left to right using a modified rejection sampling formula:
   $$\text{Acceptance Probability} = \min\left(1, \frac{P_{\text{target}}(\hat{x}_i)}{P_{\text{draft}}(\hat{x}_i)}\right)$$
4. **Correction and Bonus Token:** If token $\hat{x}_{M+1}$ is rejected, all subsequent tokens are discarded. The target model samples a replacement token directly from the adjusted residual distribution:
   $$P_{\text{corrected}}(x) = \text{normalize}\left(\max\left(0, P_{\text{target}}(x) - P_{\text{draft}}(x)\right)\right)$$
   If all $K$ tokens are accepted, the target model predicts a $(K+1)$-th **bonus token** for free.

#### Exact Mathematical Equivalence
Because the acceptance and rejection mechanics follow strict statistical rejection sampling, the generated token distribution is **provably identical** to sampling directly from the large target model alone. Speculative decoding guarantees $100\%$ lossless output with zero degradation in reasoning, perplexity, or creativity.

## Minimal example
Generating code with a 70B target model using a 1B draft model ($K = 4$ candidate tokens):
- **Draft Generation:** The 1B draft model proposes 4 tokens (`"def"`, `" foo"`, `"():"`, `"\n"`) taking $8\text{ ms}$.
- **Target Verification:** The 70B model processes all 4 tokens in parallel in a single forward pass taking $32\text{ ms}$.
- **Acceptance:** The 70B model accepts all 4 tokens and emits a bonus 5th token (`"    return"`).
- **Result:** 5 tokens produced in $40\text{ ms}$ ($8.0\text{ ms/token}$). Generating those 5 tokens with the 70B model standardly would require $5 \times 32\text{ ms} = 160\text{ ms}$ ($32\text{ ms/token}$). The system achieves an exact **$4.0\times$ speedup** with zero loss of accuracy.

## Prerequisites
- None

## Common confusions
- **Speculative decoding approximates model outputs:** Unlike lossy quantization or pruning, speculative decoding is **mathematically exact**. The sampling mathematics guarantees that the output probability distribution matches the target model to the last decimal place.
- **The draft model must be a smaller version of the same family:** While smaller checkpoints from the same family (e.g., LLaMA-1B drafting for LLaMA-70B) provide the highest acceptance rates, any model sharing the exact same tokenizer (or even specialized diffusion heads and medusa heads) can serve as a draft model.

## Appears in
- [[lessons/2026-11-01-llm-quantization-and-deployment-techniques]]: introduced as the premier lossless decoding acceleration technique for large language models.
