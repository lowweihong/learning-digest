---
title: Review questions
---

Questions from every lesson, newest first. Answers are folded, so try each question before you open it.

<!-- The daily run inserts new sections below this line: "## YYYY-MM-DD — [[lessons/<stem>|<title>]]" followed by that lesson's question callouts. -->

## 2026-11-01 — [[lessons/2026-11-01-quantization-k-means-and-linear|Neural Network Quantization: K-Means and Linear Quantization]]

> [!question]- Q1 (recall): What is the fundamental operational difference between K-means codebook quantization and linear (affine) quantization during inference?
> In K-means codebook quantization, weights are stored as integer indices and dequantized via a lookup table, meaning arithmetic operations are still executed in floating-point. In linear quantization, both storage and arithmetic operations (multiply-accumulate) are executed directly in integer arithmetic using scale factors and zero-points. Linked note: [[concepts/k-means-quantization|K-Means Quantization]] and [[concepts/linear-quantization|Linear Quantization]].

> [!question]- Q2 (explain-why): Why must the zero-point $Z$ in linear quantization be restricted to an integer value rather than a floating-point number?
> The zero-point lives in the quantized integer domain and is directly added to or subtracted from integer activations and weights inside integer registers. If $Z$ were a floating-point number, integer arithmetic operations would require conversion back to floating-point, defeating the objective of integer-only execution. Linked note: [[concepts/zero-point|Zero-Point]].

> [!question]- Q3 (apply): A weight tensor has continuous values ranging between $R_{\min} = -3.0$ and $R_{\max} = 5.0$. Compute the scale factor $S$ and zero-point $Z$ for 8-bit signed integer linear quantization ($Q_{\min} = -128, Q_{\max} = 127$).
> The scale factor is $S = \frac{R_{\max} - R_{\min}}{Q_{\max} - Q_{\min}} = \frac{5.0 - (-3.0)}{127 - (-128)} = \frac{8.0}{255} \approx 0.03137$. The zero-point is $Z = \text{round}\left(Q_{\min} - \frac{R_{\min}}{S}\right) = \text{round}\left(-128 - \frac{-3.0}{0.03137}\right) = \text{round}(-128 + 95.625) = \text{round}(-32.375) = -32$. Linked note: [[concepts/linear-quantization|Linear Quantization]].

> [!question]- Q4 (transfer): Vector quantization in signal processing (such as audio/speech compression) clusters waveform vectors into a codebook using the Linde-Buzo-Gray (LBG) algorithm. How does fine-tuning in neural network K-means quantization differ from classical codebook updates, and why can't we simply re-run standard K-means after every training batch?
> Classical vector quantization updates centroids to minimize reconstruction error (mean squared error of signals). In neural network fine-tuning, centroids are updated using task gradients from the end-to-end loss function ($\Delta c_i = -\eta \sum_{w \in C_i} \frac{\partial L}{\partial w}$) rather than Euclidean distance to original weights. Running standard K-means every batch would re-cluster weights based on current magnitudes, destroying cluster assignments and ignoring task error gradients. Linked note: [[concepts/k-means-quantization|K-Means Quantization]].
