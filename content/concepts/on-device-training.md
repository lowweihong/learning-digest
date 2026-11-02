---
title: On-Device Training
aliases: [edge training, on-device fine-tuning, continual learning on edge]
tags: [concept, machine-learning, edge-computing]
---

## Definition
Running the backpropagation and gradient update steps of neural network training directly on a local edge device (such as a phone, robot, or microcontroller) without sending user data to the cloud.

## Why it's true
Pre-trained models are trained on general internet datasets, but real-world edge devices face unique local environments—like a smart doorbell adjusting to a specific family's voices or a robotic arm adapting to a new tool. Sending private audio or video to the cloud compromises user privacy. However, training requires saving intermediate activations for backpropagation, typically needing 3× more compute and hundreds of megabytes of memory. On-device training techniques make learning fit within tiny memory budgets by only updating select sparse layers and quantizing intermediate gradients.

## Minimal example
An ARM Cortex-M7 board has two buttons (Green = "No Person", Red = "Person") and a camera. Initially, the model has random weights and guesses randomly. A user points the camera at a person, presses the Red button, points at an empty wall, and presses the Green button. The chip executes backpropagation locally using less than **256 KB of memory**, adapting its weights within seconds without Wi-Fi or cloud servers.

## Prerequisites
- [[concepts/tinyml|TinyML]]

## Common confusions
- Assuming training is just inference run in reverse: Inference only needs forward passes and discards intermediate values as soon as a layer finishes. Training must store all layer activations to calculate gradients during the backward pass, making memory management drastically harder.

## Appears in
- [[lessons/2026-11-01-intro-to-tinyml-and-efficient-ai]]: demonstrated as the frontier beyond inference for privacy-preserving, lifelong edge learning.
