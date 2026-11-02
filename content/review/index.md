---
title: Review questions
---

Questions from every lesson, newest first. Answers are folded, so try each question before you open it.

<!-- The daily run inserts new sections below this line: "## YYYY-MM-DD — [[lessons/<stem>|<title>]]" followed by that lesson's question callouts. -->

## 2026-11-01 — [[lessons/2026-11-01-quantization-k-means-and-linear|Neural Network Quantization: From Physical Bits to Integer Arithmetic]]

> [!question]- Q1 (recall): Why does K-means weight sharing reduce memory traffic but fail to speed up arithmetic on low-power chips without an FPU?
> K-means stores weights as compact integer bucket IDs, saving memory transfer bandwidth. However, during computation, each ID must be looked up in a table to fetch the original 32-bit decimal centroid, meaning the processor still has to perform full floating-point arithmetic. Linked note: [[concepts/k-means-quantization|K-Means Quantization]].

> [!question]- Q2 (explain-why): Why must the zero-point $Z$ in linear quantization be an exact integer rather than a decimal number?
> The zero-point lives in the integer domain and is directly added to or subtracted from integer values inside hardware registers. If $Z$ were a decimal, the processor would be forced to convert everything back into floating-point math, defeating the goal of running pure integer arithmetic. Linked note: [[concepts/zero-point|Zero-Point]].

> [!question]- Q3 (apply): A layer has decimal values between $R_{\min} = -2.0$ and $R_{\max} = 6.0$. Calculate the scale factor $S$ and zero-point $Z$ for standard signed 8-bit integers ($Q_{\min} = -128, Q_{\max} = 127$).
> The scale factor is $S = \frac{R_{\max} - R_{\min}}{Q_{\max} - Q_{\min}} = \frac{6.0 - (-2.0)}{127 - (-128)} = \frac{8.0}{255} \approx 0.03137$. The zero-point is $Z = \text{round}\left(Q_{\min} - \frac{R_{\min}}{S}\right) = \text{round}\left(-128 - \frac{-2.0}{0.03137}\right) = \text{round}(-128 + 63.75) = \text{round}(-64.25) = -64$. Linked note: [[concepts/linear-quantization|Linear Quantization]].

> [!question]- Q4 (transfer): In voice compression, Vector Quantization clusters sound waves into codebook buckets. How does neural network K-means fine-tuning differ from standard audio clustering, and what would happen if you re-clustered weights with standard K-means after every training batch?
> Standard clustering updates centroids to minimize geometric distance to data points. Neural network fine-tuning updates centroids using task gradients from backpropagation ($\Delta c_k = -\eta \sum \frac{\partial L}{\partial w}$) to minimize task loss. Re-clustering with standard K-means every batch would randomly shuffle weights into different buckets based on distance, destroying training momentum and ignoring whether predictions are actually accurate. Linked note: [[concepts/k-means-quantization|K-Means Quantization]].
