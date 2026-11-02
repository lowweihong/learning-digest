---
title: QLoRA
aliases: [QLoRA, quantized LoRA, 4-bit fine-tuning, NormalFloat4, double quantization]
tags: [concept, llm, peft, quantization, efficient-ai]
---

## Definition
An extreme parameter-efficient fine-tuning framework that quantizes the frozen pre-trained base model down to 4-bit precision (**NormalFloat4**) while training a small 16-bit [[concepts/low-rank-adaptation|LoRA]] adapter, utilizing **Double Quantization** and **Paged Optimizers** to enable fine-tuning 70B parameter foundation models on a single commercial GPU.

## Why it's true
While standard LoRA reduces trainable parameter memory and optimizer states by $99\%$, the frozen pre-trained base model weights must still reside in GPU VRAM in full 16-bit precision. For a 70B model, holding the frozen FP16 weights alone requires $140\text{ GB}$ of VRAM, mandating expensive multi-GPU clusters (e.g., two 80 GB A100s) simply to host the base model.

**QLoRA (Quantized LoRA)** resolves this through three algorithmic and architectural innovations:
1. **4-bit NormalFloat (NF4):** Neural network weights trained with gradient descent follow a zero-mean Gaussian distribution $\mathcal{N}(0, \sigma^2)$. Standard 4-bit integer or floating-point formats distribute quantization bins uniformly, which is suboptimal for bell curves. NF4 creates non-uniform quantization intervals whose bins each hold an equal probability mass under a normal distribution, establishing an information-theoretically optimal 4-bit representation.
2. **Double Quantization (DQ):** In block-wise quantization (e.g., block size 64), each group requires a 32-bit floating-point quantization constant, adding $32 / 64 = 0.5$ bits per parameter. Double Quantization treats these scaling constants as numbers and quantizes them with an 8-bit FP8 quantizer, reducing the scale footprint from $0.5$ bits down to $0.127$ bits per parameter (saving $3\text{ GB}$ on a 65B model).
3. **Paged Optimizers:** Uses CUDA Unified Memory to automatically page optimizer state tensors between GPU VRAM and CPU system RAM during temporary gradient checkpointing spikes, eliminating out-of-memory (OOM) hard faults.

During the forward pass, 4-bit NF4 weights are dequantized on-the-fly into 16-bit BrainFloat (BF16) in registers to compute matrix products, while gradients are computed and backpropagated strictly into the high-precision 16-bit LoRA adapters.

## Minimal example
Hardware requirements for fine-tuning a 65B/70B parameter foundation model:
- **16-bit Full Fine-Tuning:**
  Model ($140\text{ GB}$) + Gradients ($140\text{ GB}$) + Adam States ($1.12\text{ TB}$) $\implies \mathbf{> 780\text{ GB}}$ VRAM (requires at least eight 80 GB A100 GPUs).
- **16-bit Standard LoRA:**
  Frozen Model ($140\text{ GB}$) + LoRA states ($\approx 10\text{ GB}$) $\implies \mathbf{> 160\text{ GB}}$ VRAM (requires at least two 80 GB A100 GPUs).
- **QLoRA (4-bit NF4):**
  Frozen Model compressed to $3.5\text{ bits/weight}$ ($\approx 32\text{ GB}$) + LoRA states ($\approx 10\text{ GB}$) $\implies \mathbf{\approx 42\text{ GB}}$ VRAM.
  The entire 70B training process executes on a **single 48 GB workstation GPU** (NVIDIA RTX 6000 Ada or A6000), achieving benchmark performance parity with 16-bit full fine-tuning.

## Prerequisites
- [[concepts/low-rank-adaptation|Low-Rank Adaptation]], [[concepts/linear-quantization|Linear Quantization]]

## Common confusions
- **QLoRA quantizes the adapter weights:** No. The LoRA adapter matrices ($A$ and $B$) are trained in full **16-bit precision** (FP16 or BF16) to ensure numerical stability and high dynamic range during gradient backpropagation. Only the frozen pre-trained base model is compressed to 4-bit NF4.
- **QLoRA is slower than standard LoRA:** Because 4-bit NF4 weights must be dequantized to 16-bit on-the-fly during forward and backward passes, QLoRA is roughly $20\text{–}30\%$ slower in raw step clock time than 16-bit LoRA, but enables training on hardware that would otherwise be physically incapable of running the workload.

## Appears in
- [[lessons/2026-11-01-llm-post-training-and-alignment]]: introduced as the breakthrough quantization-adaptation pipeline democratizing open-source LLM training.
