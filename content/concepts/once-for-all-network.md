---
title: Once-for-All Network
aliases: [OFA, Once-for-All, progressive shrinking]
tags: [concept, automl, nas, efficient-ai]
---

## Definition
A super-network training paradigm where a single over-parameterized model is trained once using a progressive shrinking schedule to support an astronomical number of sub-networks (e.g., $10^{19}$) that can be extracted instantaneously for diverse hardware targets with zero retraining.

## Why it's true
In real-world deployment, software must target hundreds of distinct hardware profiles (cloud GPUs, desktop cards, laptops, mobile phones, microcontrollers). Re-running NAS and retraining models from scratch for every individual chip creates an unsustainable computational cost ($N \times \text{training cost}$).

Once-for-All (OFA) decouples model training from architecture search:
1. **Train Once:** Train a single super-network supporting elastic dimensions (depth, width, kernel size, input resolution).
2. **Progressive Shrinking:** Start training the largest network to convergence, then progressively enforce smaller sub-networks (using weight sharing and kernel transformation matrices) so smaller networks inherit trained weights from larger counterparts without catastrophic forgetting.
3. **Specialize for All:** Deploy to any device in seconds by running a lightweight genetic search over the pre-trained supernet to find the Pareto-optimal sub-network for that specific hardware latency budget.

## Minimal example
Deploying an ImageNet model across a datacenter GPU, an iPhone, and a Cortex-M7 microcontroller:
- **Traditional NAS:** 3 independent searches and full training runs $\rightarrow 3 \times 1,200\text{ GPU hours} = 3,600\text{ GPU hours}$.
- **OFA:** Train the OFA supernet once ($1,200\text{ GPU hours}$). Then, run an evolutionary search against pre-built latency lookup tables for 10 minutes per device.
  - H100 GPU extracts a wide, shallow sub-network with $7 \times 7$ kernels.
  - iPhone extracts a balanced $5 \times 5$ sub-network.
  - Cortex-M7 extracts a low-depth, small-kernel sub-network fitting $256\text{ KB}$ SRAM.
All three models run immediately with zero fine-tuning.

## Prerequisites
- [[concepts/neural-architecture-search|Neural Architecture Search]]

## Common confusions
- **Weight interference during training:** If random sub-networks are sampled naively from day one, small sub-networks corrupt the weights of larger networks. OFA's **Progressive Shrinking (PS)** solves this by training largest-to-smallest in stages: full network $\to$ elastic kernel size ($7 \times 7 \to 5 \times 5 \to 3 \times 3$) $\to$ elastic depth $\to$ elastic width.

## Appears in
- [[lessons/2026-11-01-hardware-aware-nas-and-once-for-all-networks]]: introduced as the foundation for multi-device deployment at near-zero search cost.
