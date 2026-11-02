---
title: Low-Rank Adaptation
aliases: [LoRA, low-rank fine-tuning, PEFT LoRA, parameter-efficient adaptation]
tags: [concept, llm, peft, post-training, fine-tuning]
---

## Definition
A parameter-efficient fine-tuning (**PEFT**) technique that freezes the pre-trained weight matrices of a neural network ($W_0$) and models task-specific weight updates through low-rank matrix decomposition: $\Delta W = B \cdot A$ (where rank $r \ll \min(d, k)$), reducing trainable parameters by up to $10,000\times$ and introducing zero additional inference latency.

## Why it's true
Full parameter fine-tuning of modern foundation models (e.g., 70B parameters) incurs prohibitive storage and computational costs:
1. **Optimizer Memory Bloat:** Training with the Adam optimizer requires storing the first and second gradient moments for every single parameter in FP32, requiring $16$ bytes of optimizer state per weight ($1.12\text{ TB}$ of GPU memory for a 70B model).
2. **Multi-Task Storage Explosion:** Deploying specialized models for 100 enterprise clients requires storing 100 complete model checkpoints ($100 \times 140\text{ GB} = 14\text{ Terabytes}$).

**Low-Rank Adaptation (LoRA)** exploits the principle of **intrinsic dimension**: Aghajanyan et al. proved that despite having billions of parameters, the actual weight trajectory $\Delta W$ required to adapt a foundation model to a specific task resides on a low-dimensional manifold.

LoRA decomposes the weight update into two low-rank matrices:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
where:
- $W_0 \in \mathbb{R}^{d \times k}$ is the frozen pre-trained weight.
- $A \in \mathbb{R}^{r \times k}$ projects the input down to low rank $r$ (e.g., $r = 8$ or $16$).
- $B \in \mathbb{R}^{d \times r}$ projects the low-rank representation back up to the original output dimension.
- $\alpha$ is a constant scaling hyperparameter.

#### Initialization and Zero Inference Overhead
- **Initialization:** $A$ is initialized with a Gaussian random distribution, and $B$ is initialized to **zero**. At step $0$, $\Delta W = B \cdot A = 0$, ensuring the model's pre-trained behavior is $100\%$ preserved at the start of training.
- **Zero Inference Latency:** Unlike standard bottleneck adapters (which add sequential layers into the computational graph and increase runtime latency), LoRA updates can be algebraically added directly into the base weights prior to deployment:
  $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
  The deployed model runs with identical architecture, identical memory layout, and **zero extra latency**.

## Minimal example
A linear projection layer in LLaMA with input dimension $d = 4,096$ and output dimension $k = 4,096$:
- **Full Fine-Tuning Parameters:**
  $$4,096 \times 4,096 = 16,777,216 \text{ parameters}$$
- **LoRA Parameters ($r = 8$):**
  $$\text{Matrix } A: 8 \times 4,096 = 32,768, \quad \text{Matrix } B: 4,096 \times 8 = 32,768 \implies \text{Total} = 65,536 \text{ parameters}$$
Trainable parameters shrink by **$256\times$** ($99.6\%$ reduction), allowing fine-tuning on a single GPU.

## Prerequisites
- None

## Common confusions
- **LoRA only applies to self-attention:** While early papers applied LoRA primarily to Query ($W_q$) and Value ($W_v$) projections, modern practice applies LoRA across **all linear projections**, including Key ($W_k$), Output ($W_o$), and MLP layers ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$), yielding superior representational quality at negligible added parameter cost.
- **LoRA adapters must be merged permanently:** In multi-tenant cloud serving, systems (like S-LoRA or Punica) keep the base model frozen and dynamically swap lightweight LoRA matrices ($A$ and $B$) on-the-fly, serving hundreds of customized client models from a single GPU instance.

## Appears in
- [[lessons/2026-11-01-llm-post-training-and-alignment]]: introduced as the foundational parameter-efficient fine-tuning method across language and diffusion models.
