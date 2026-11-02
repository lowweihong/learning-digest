---
title: SmoothQuant
aliases: [W8A8 LLM quantization, activation smoothing, channel migration quantization]
tags: [concept, llm, quantization, efficient-ai, systems]
---

## Definition
A post-training quantization framework that enables 8-bit weight and 8-bit activation (**W8A8**) INT8 inference across large language models by migrating quantization difficulty from spiky activation outlier channels into smooth, quantization-friendly weight channels through an exact mathematical equivalence.

## Why it's true
While weights of large language models are relatively easy to quantize to 8-bit integers, activations present a formidable obstacle known as the **activation outlier phenomenon**:
1. **Emergence of Outliers:** Once transformer models surpass $6.7\text{B}$ parameters, specific feature channels begin producing activation values whose magnitudes ($70\text{–}100$) are up to $100\times$ larger than average tokens ($0.1\text{–}0.5$).
2. **Dynamic Range Collapse:** In standard symmetric INT8 quantization with 256 discrete levels ($[-128, 127]$), the scale factor $\Delta = \frac{\max |X|}{127}$ is dictated entirely by the outlier. Small, subtle activations are rounded down to zero, destroying the model's linguistic and reasoning capabilities.
3. **Weight vs. Activation Asymmetry:** While activations have extreme outliers, weight matrices are statically distributed and exhibit smooth, bell-shaped distributions with no extreme spikes.

**SmoothQuant** exploits this asymmetry using an exact linear equivalence:
$$Y = X W = \left( X \cdot \text{diag}(s)^{-1} \right) \cdot \left( \text{diag}(s) \cdot W \right) = \hat{X} \hat{W}$$
- For each input channel $j$, calculate a per-channel smoothing scale $s_j$ that balances activation magnitude against weight magnitude:
  $$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$
  where $\alpha \in [0, 1]$ is a migration strength hyperparameter (typically $\alpha = 0.5$).
- **Dividing activations by $s_j$** shrinks the activation outlier down into a manageable range.
- **Multiplying weights by $s_j$** inflates the corresponding weight values, but because weights were previously small, they easily absorb the increase within INT8 boundaries.
- **Zero Runtime Overhead:** The inverse scale $\text{diag}(s)^{-1}$ is folded offline into the scale and bias parameters of the preceding LayerNorm or linear operation. The model executes purely on high-throughput INT8 Tensor Cores without adding extra CUDA kernels or runtime latency.

## Minimal example
A linear layer channel where activation peaks at $64.0$ while corresponding weights peak at $1.0$:
- **Before Smoothing:** Quantizing activations to INT8 forces scale factor $\Delta_X = 64 / 127 \approx 0.504$. Normal tokens with values $\le 0.25$ round to zero, causing output perplexity to explode to $>40$ (unusable).
- **SmoothQuant ($\alpha = 0.5$):**
  $$s = \sqrt{\frac{64}{1}} = 8$$
  - Scaled activation peak: $\hat{X} = 64 / 8 = 8.0$.
  - Scaled weight peak: $\hat{W} = 1.0 \times 8 = 8.0$.
- **After Smoothing:** Both activation ($[-8, 8]$) and weight ($[-8, 8]$) channels comfortably map onto INT8 ($[-128, 127]$). Output perplexity remains identical to FP16 baseline ($10.2$), while inference latency drops by $2\times$ on INT8 Tensor Cores.

## Prerequisites
- [[concepts/linear-quantization|Linear Quantization]]

## Common confusions
- **SmoothQuant requires retraining:** SmoothQuant is strictly a post-training quantization (PTQ) technique. Smoothing scales are calculated using a tiny calibration set of 32–64 sample prompts in minutes, requiring zero backpropagation or dataset re-labeling.
- **SmoothQuant is weight-only:** SmoothQuant targets **both weights and activations (W8A8)** to leverage INT8 matrix-multiply tensor cores, whereas weight-only quantization (like AWQ) leaves activations in FP16.

## Appears in
- [[lessons/2026-11-01-llm-quantization-and-deployment-techniques]]: introduced as the breakthrough technique enabling W8A8 INT8 serving in large language models.
