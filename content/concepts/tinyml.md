---
title: TinyML
aliases: [Tiny AI, microcontroller ML, edge AI]
tags: [concept, embedded-systems, edge-computing]
---

## Definition
Running machine learning inference and training on extremely resource-constrained microcontrollers (MCUs) with only hundreds of kilobytes of memory and milliwatts of power consumption.

## Why it's true
Cloud AI requires sending private camera, microphone, or sensor data across the internet to giant data centers, creating privacy risks, network latency, and high electricity bills. Microcontrollers (like the ARM Cortex-M family found in thermostats, appliances, and cars) cost a few dollars and use almost zero battery. If models can be compressed to fit their tiny on-chip memory (SRAM), devices can see, hear, and decide locally without an internet connection.

## Minimal example
MCUNet shrinks a full convolutional vision model to fit an ARM Cortex-M7 microcontroller with just **256 KB of SRAM** and **1 MB of Flash storage**. Running on a few milliwatts of battery, it continuously detects human faces and draws bounding boxes in real time without sending a single byte to the cloud.

## Prerequisites
- [[concepts/algorithm-hardware-co-design|Algorithm-Hardware Co-Design]]

## Common confusions
- Assuming edge AI only means smartphones: Smartphones have gigabytes of memory and dedicated neural processing chips. TinyML targets devices with 10,000× less memory (kilobytes instead of gigabytes) and zero cooling fans.

## Appears in
- [[lessons/2026-11-01-intro-to-tinyml-and-efficient-ai]]: introduced as the extreme frontier of resource-constrained AI deployment.
