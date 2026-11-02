---
title: Soft Targets and Temperature
aliases: [temperature scaling, soft logits, Hinton distillation loss]
tags: [concept, deep-learning, distillation, mathematics]
---

## Definition
A mathematical transformation that softens extreme probability spikes in neural network outputs by dividing pre-softmax logits by a temperature parameter $T > 1$, turning near-binary classifications into smooth distributions that expose the relative likelihoods of non-target classes.

## Why it's true
In modern neural networks, the standard softmax function:
$$q_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$
amplifies differences between logits through exponentiation. If the top logit is $z_1 = 12$ and the runner-up is $z_2 = 4$, the resulting probabilities are roughly $0.9996$ and $0.0003$. Virtually all gradients for incorrect classes vanish into floating-point rounding errors, hiding the teacher's knowledge about which non-target classes are plausible.

By introducing temperature $T$:
$$q_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
1. **At $T = 1$:** Recovers the standard, sharp softmax output.
2. **At $T > 1$:** Smooths out the distribution, inflating small probabilities so the relative differences between negative classes become easily measurable numbers.
3. **At $T \to \infty$:** The distribution approaches a completely uniform distribution where all classes have probability $1/K$.

To ensure gradients from the softened distillation loss match the scale of standard cross-entropy loss, the KL divergence term is multiplied by $T^2$:
$$\lim_{T \to \infty} \frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^{(S)}} \approx \frac{1}{2 T^2} (z_i^{(S)} - z_i^{(T)})$$
Multiplying by $T^2$ prevents the distillation gradient from vanishing as higher temperatures flatten the logits.

## Minimal example
Consider an image of a motorcycle evaluated by a teacher model with output logits $z = [10.0, 5.0, 0.0]$ corresponding to `[Motorcycle, Bicycle, Aeroplane]`:
- **At $T = 1$:**
  $$q = [0.9932,\; 0.0067,\; 0.000045]$$
  The bicycle probability is tiny, and aeroplane is near zero; the student receives almost no signal distinguishing whether two wheels or wings are closer to a motorcycle.
- **At $T = 5$:**
  Logits become $z / 5 = [2.0, 1.0, 0.0]$:
  $$q^{(5)} = [0.665,\; 0.245,\; 0.090]$$
  Now, the bicycle receives $24.5\%$ probability and the aeroplane receives $9.0\%$. The student network directly absorbs the fact that a motorcycle is roughly $2.7\times$ more structurally related to a bicycle than to an aeroplane.

## Prerequisites
- [[concepts/knowledge-distillation|Knowledge Distillation]]

## Common confusions
- **Using $T > 1$ at inference time:** Temperature scaling is applied strictly during training to calculate loss gradients. When the student model is deployed in production, temperature is reset to $T = 1$ to produce crisp, calibrated predictions.
- **Omitting the $T^2$ scaling factor:** Forgetting to multiply the distillation loss by $T^2$ causes the gradient magnitude to shrink by $1/T^2$, rendering the teacher's supervision ineffective when combined with standard cross-entropy.

## Appears in
- [[lessons/2026-11-01-knowledge-distillation-principles]]: introduced as the core mathematical formulation of Hinton's knowledge distillation.
