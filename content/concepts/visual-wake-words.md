---
title: Visual Wake Words
aliases: [VWW, visual wake word, edge person detection]
tags: [concept, tinyml, computer-vision, efficient-ai, edge-computing]
---

## Definition
A benchmark dataset and TinyML vision classification task designed to detect whether at least one person is present in an image frame, serving as a low-power, always-on visual trigger analogous to spoken audio wake words (like "Hey Siri" or "OK Google").

## Why it's true
Modern smart cameras and IoT monitors often transmit continuous video feeds to cloud servers for facial recognition or multi-class object detection. This architecture has three crippling flaws:
1. **Severe Power Consumption:** Continuous Wi-Fi/cellular video streaming drains batteries in hours, making self-powered edge sensors impossible.
2. **Network Dependency and Latency:** Disconnected or remote environments cannot guarantee broadband upload speeds.
3. **Severe Privacy Hazards:** Transmitting raw video into cloud databases exposes sensitive personal spaces (homes, hospital rooms, offices) to potential breaches.

**Visual Wake Words (VWW)** solves this by treating vision as a binary gating mechanism:
- Formulated from the COCO dataset, an image is labeled positive ($1$) if it contains at least one bounding box for a person covering more than $2.5\%$ of the total image area, and negative ($0$) otherwise.
- The model runs continuously on a cheap, sub-dollar microcontroller consuming only milliwatts of power.
- While no person is present, the rest of the system (displays, heavy computing units, cloud transmitters) remains in deep sleep mode. When a person steps into view, the VWW model triggers the system awake.
- Because person detection relies on spatial silhouette cues, input resolution directly impacts accuracy. High-resolution input (enabled by patch-based inference) prevents missed detections for small, distant persons.

## Minimal example
An automated elevator energy-saving system:
- **Cloud Streaming Setup:** A camera streams $1080\text{p}$ video to an AWS server consuming $8\text{ W}$ of power, requiring constant network connectivity, and creating passenger surveillance concerns.
- **TinyML VWW Setup:** An off-the-shelf STM32 microcontroller paired with an ultra-low-power image sensor executes an MCUNet VWW binary classifier consuming under $20\text{ mW}$. When passengers enter the elevator, the model detects person presence in real time, illuminating the cabin. No images are stored or transmitted.

## Prerequisites
- [[concepts/tinyml|TinyML]]

## Common confusions
- **VWW performs face recognition:** VWW does not identify *who* the person is, nor does it locate facial features. It only predicts a binary label: person present ($1$) or absent ($0$).
- **VWW can use arbitrarily low resolutions:** While classifying large nearby objects works at $96 \times 96$, recognizing a person standing across a warehouse or large room requires at least $160 \times 160$ to $224 \times 224$ resolution to preserve body proportions.

## Appears in
- [[lessons/2026-11-01-tinyml-mcunet-and-patch-inference]]: introduced as the standard real-world TinyML vision benchmark for evaluating microcontroller efficiency.
