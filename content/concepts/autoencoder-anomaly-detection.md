---
title: Autoencoder Anomaly Detection
aliases: [unsupervised anomaly detection, reconstruction error detection, edge anomaly detection]
tags: [concept, tinyml, anomaly-detection, time-series, unsupervised-learning]
---

## Definition
An unsupervised learning method that trains a neural network autoencoder to compress and reconstruct normal system signals through a low-dimensional bottleneck, detecting faults and anomalies at runtime by measuring when the reconstruction error exceeds an established threshold.

## Why it's true
In industrial predictive maintenance and edge health monitoring (such as factory pumps, wind turbines, vehicle engines, and wearable ECG monitors), supervised classification cannot be applied:
1. **Severe Class Imbalance:** Equipment operates normally $99.99\%$ of the time. Failure data is exceptionally rare.
2. **Infinite Anomaly Modes:** A machine can fail in hundreds of unpredictable ways (bearing wear, blade cracking, shaft misalignment, foreign object obstruction). Collecting labeled training data for every possible failure mode is impossible and prohibitively expensive.

**Autoencoder Anomaly Detection** solves this through one-class unsupervised learning:
- **The Information Bottleneck:** An encoder $f_\theta$ compresses the multi-sensor input $x$ into a lower-dimensional latent representation $z$, and a decoder $g_\phi$ attempts to reconstruct the original input: $\hat{x} = g_\phi(f_\theta(x))$.
- **Training on Normal Data Only:** The model is trained exclusively on normal operating cycles to minimize reconstruction loss:
  $$\mathcal{L} = \| x - \hat{x} \|_2^2$$
  The bottleneck forces the network to learn only the underlying regularities and correlations of healthy operation.
- **Runtime Inference:** When deployed on a microcontroller, the device continuously computes the reconstruction error:
  $$\text{Anomaly Score} = \| x - \hat{x} \|_2^2$$
  - If the input is normal, the autoencoder reconstructs it accurately (low score).
  - If a fault or foreign obstruction occurs, the unfamiliar frequencies and dynamics cannot be reconstructed through the bottleneck, causing the reconstruction error to spike above a detection threshold $\tau$ and firing a local alarm.

## Minimal example
Monitoring vibration on an industrial cooling fan using a 3-axis accelerometer connected to a microcontroller:
- **Healthy Operation:** Fan rotates smoothly. Input accelerometer signals reconstruct cleanly with mean squared error $\text{MSE} \approx 0.02$.
- **Mechanical Fault:** A technician touches a screwdriver to the fan blades or a bearing misaligns, creating high-frequency rattling.
- **Detection:** The autoencoder has never seen this frequency distribution; reconstructed output $\hat{x}$ fails completely, causing error to spike to $\text{MSE} = 0.48$.
- **Alert:** The microcontroller compares $\text{MSE} = 0.48 > \tau = 0.10$ and immediately triggers a yellow warning LED, shutting off the motor before catastrophic destruction occurs.

## Prerequisites
- [[concepts/tinyml|TinyML]]

## Common confusions
- **Autoencoder memorizes everything:** If the latent bottleneck dimension is too wide, the autoencoder will act as an identity function, perfectly reconstructing anomalous inputs and failing to detect faults. The bottleneck must be sufficiently narrow to force generalization only on normal patterns.
- **Requires anomaly labels during training:** The training process requires zero anomaly samples. It is purely unsupervised and trained solely on healthy baseline telemetry.

## Appears in
- [[lessons/2026-11-01-tinyml-mcunet-and-patch-inference]]: introduced as the core unsupervised technique for edge time-series and predictive maintenance on microcontrollers.
