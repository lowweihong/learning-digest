---
title: SwiGLU Activation
aliases: [SwiGLU, Swish Gated Linear Unit, gated MLP, modern transformer FFN]
tags: [concept, llm, transformers, deep-learning, activations]
---

## Definition
A gated non-linear activation unit and feed-forward block used in modern transformer architectures that combines linear transformations with a Swish-activated gating branch via element-wise multiplication: $\text{SwiGLU}(x) = \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}})$, followed by a down-projection matrix.

## Why it's true
In the original Transformer architecture, the Feed-Forward Network (FFN) was a standard two-layer perceptron with a single non-linear activation (ReLU or GELU):
$$\text{FFN}_{\text{standard}}(x) = \text{Activation}(x W_1 + b_1) W_2 + b_2$$
where $W_1$ expands the hidden state from $D$ to $4D$, and $W_2$ projects it back from $4D$ to $D$, requiring $2 \times 4 D^2 = 8 D^2$ parameters.

Research by Noam Shazeer (2020) demonstrated that **Gated Linear Units (GLUs)** provide substantially stronger representational capacity:
1. **Multiplicative Information Gating:** Instead of passing the entire activation through a fixed non-linearity, a GLU splits the transformation into two parallel paths: a content branch ($x W_{\text{up}}$) and a gating branch ($x W_{\text{gate}}$). The gating branch dynamically scales how much information passes through each coordinate.
2. **Smooth Gradient Flow:** By pairing the gate with the smooth Swish activation ($\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$), SwiGLU avoids the "dead neuron" pathology of ReLU where negative inputs produce exactly zero gradients.
3. **Parameter Calibration to $\frac{8}{3} D$:** Because SwiGLU utilizes three projection matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) instead of two, keeping the intermediate width at $4D$ would inflate parameters by $50\%$ ($12 D^2$). Modern LLMs calibrate the intermediate dimension $d_{\text{ffn}}$ to approximately $\frac{8}{3} D$:
   $$\text{Total Parameters} = 3 \times \left( \frac{8}{3} D \times D \right) = 8 D^2$$
   This ensures that SwiGLU preserves the exact same parameter budget and FLOP count as a traditional $4D$ FFN while delivering strictly superior perplexity and reasoning scores.

## Minimal example
Implementing SwiGLU in a foundation model with hidden dimension $D = 4,096$ (as in LLaMA-7B):
- Calculate intermediate dimension: $d_{\text{ffn}} = \text{round}\left(\frac{8}{3} \times 4096\right) = 11,008$.
- Forward pass:
  ```python
  gate = x @ W_gate  # Shape: (batch, seq, 11008)
  up = x @ W_up  # Shape: (batch, seq, 11008)
  hidden = (gate * torch.sigmoid(gate)) * up  # SwiGLU element-wise gating
  output = hidden @ W_down  # Shape: (batch, seq, 4096)
```
The architecture provides multiplicative filtering across all 11,008 intermediate dimensions at zero parameter overhead.

## Prerequisites
- None

## Common confusions
- **SwiGLU increases inference memory:** While SwiGLU introduces a third weight matrix per layer, its intermediate channel dimension is reduced from $4D$ down to $\frac{8}{3} D$, keeping activation memory and FLOP counts virtually identical to standard FFN blocks.
- **SwiGLU replaces Self-Attention:** SwiGLU operates strictly within the token-level Feed-Forward Network (FFN). It mixes features *within* each token along the channel dimension, whereas attention mixes information *across* different tokens along the sequence dimension.

## Appears in
- [[lessons/2026-11-01-transformers-llms-and-attention-mechanisms]]: introduced as the modern feed-forward block standard in LLaMA, Mistral, and contemporary LLMs.
