---
title: Rotary Position Embedding
aliases: [RoPE, rotary position encoding, complex plane position rotation, position interpolation]
tags: [concept, llm, transformers, mathematics, attention]
---

## Definition
A relative positional encoding technique that encodes token positions by rotating pairs of coordinates in Query and Key vectors in the 2D complex plane by an angle proportional to the token's sequence index, guaranteeing that the attention dot product between two tokens depends strictly on their relative distance.

## Why it's true
In self-attention, the raw dot product $q^T k$ is permutation-invariant: it contains zero inherent knowledge of word order.

Earlier models addressed this using **Absolute Position Embeddings** ($x_m + p_m$). However, absolute embeddings suffer from two crippling flaws:
1. They add position offsets only to the initial input layer, which gradually decays across deep transformer stacks.
2. They cannot extrapolate: if a model is pre-trained with a maximum context window of $2,048$ tokens, it possesses no embeddings for position $2,049$ and collapses completely on longer documents.

**Rotary Position Embedding (RoPE)** resolves this geometrically:
1. **2D Coordinate Pairing:** Divide each $D$-dimensional query and key vector into $D/2$ orthogonal two-dimensional pairs: $(x_1, x_2), (x_3, x_4), \dots, (x_{D-1}, x_D)$.
2. **Complex Plane Rotation:** Treat each 2D pair as a complex number and rotate it by an angle $m \theta_i$, where $m$ is the absolute sequence index and $\theta_i = 10000^{-2(i-1)/D}$:
   $$R_{\Theta, m}^{(i)} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}$$
3. **Relative Dot-Product Invariance:** When calculating the attention score between query $q_m$ at position $m$ and key $k_n$ at position $n$:
   $$\langle R_{\Theta, m} q_m, R_{\Theta, n} k_n \rangle = q_m^T R_{\Theta, m}^T R_{\Theta, n} k_n = q_m^T R_{\Theta, n - m} k_n$$
   The absolute positions $m$ and $n$ cancel out. The resulting attention weight is a direct mathematical function of the **relative distance $(n - m)$**.

#### Context Length Extension via Position Interpolation
Because RoPE represents position as rotational frequency angles, extending a model's context window (e.g., from $2\text{K}$ to $32\text{K}$ tokens) does not require learning new positional embeddings. By applying **Position Interpolation (PI)**, developers downscale position indices by a ratio $S = 32\text{K} / 2\text{K} = 16$:
$$m' = \frac{m}{S}$$
This compresses the angles so all 32K positions fall within the familiar rotational angle boundaries observed during pre-training, enabling long-context fine-tuning in just a few hundred steps.

## Minimal example
A 2D query at token position $m = 7$ ($q_7$) and a key at token position $n = 3$ ($k_3$):
- $q_7$ is rotated by angle $7\theta$.
- $k_3$ is rotated by angle $3\theta$.
- Taking their dot product:
  $$\langle R(7\theta) q_7, R(3\theta) k_3 \rangle = q_7^T R(7\theta - 3\theta) k_3 = q_7^T R(4\theta) k_3$$
The resulting score reflects the exact relative separation of 4 tokens, regardless of whether the pair occurs at the beginning of the text or 10,000 tokens later.

## Prerequisites
- None

## Common confusions
- **RoPE multiplies dense rotation matrices at runtime:** Although formulated as matrix multiplication, RoPE is implemented in practice with vector element-wise multiplications, coordinate swaps, and sign negations, adding almost zero computational overhead:
  $$R_{\Theta, m} x = x \odot \cos(m \Theta) + \tilde{x} \odot \sin(m \Theta)$$
- **RoPE is applied to Value vectors:** RoPE is applied strictly to **Query and Key** vectors to determine attention weights. It is never applied to Value vectors ($V$), preserving raw semantic token content during aggregation.

## Appears in
- [[lessons/2026-11-01-transformers-llms-and-attention-mechanisms]]: introduced as the predominant relative position encoding mechanism in modern foundation models (LLaMA, Mistral, Qwen).
