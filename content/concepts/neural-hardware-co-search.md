---
title: Neural-Hardware Co-Search
aliases: [NAAS, hardware-software co-search, joint NAS-accelerator search]
tags: [concept, hardware, automl, nas]
---

## Definition
An end-to-end automated optimization framework that jointly searches three tightly coupled design spaces simultaneously: the neural network architecture, the compiler mapping strategy (loop order and tiling), and the physical accelerator microarchitecture (buffer sizes and processing element array connectivity).

## Why it's true
Designing neural networks for a fixed, off-the-shelf chip leaves significant efficiency on the table. Conversely, designing custom silicon ASICs for an existing neural network risks rapid obsolescence when models evolve.

Neural-Hardware Co-Search unifies all three layers:
- **Neural Network Space:** Layer depths, channel widths, kernel sizes, and attention types.
- **Compiler Mapping Space:** Loop order (which loop executes outermost vs. innermost), loop tiling sizes, and spatial vs. temporal parallel execution.
- **Accelerator Architecture Space:** On-chip SRAM buffer sizes (input, weight, output buffers), processing element (PE) array dimensions ($18 \times 10$), and inter-PE interconnect routing.

By encoding non-numerical loop orders and connectivity into continuous importance-based ranking vectors, an evolutionary search explores all three spaces in a single optimization loop to maximize the Energy-Delay Product (EDP).

## Minimal example
Designing an edge AI vision accelerator:
- **Hardware-only search:** Sizing buffers on a baseline ResNet gives $1.3\times$ energy-delay reduction.
- **Neural-Hardware Co-Search (NAAS):** Discovers an asymmetric network paired with an $18 \times 10$ PE array and a custom dataflow that parallelizes output height and channel dimensions together.
The combined design delivers a **$4.2\times$ reduction in Energy-Delay Product** while boosting ImageNet accuracy by $2.7\%$.

## Prerequisites
- [[concepts/neural-architecture-search|Neural Architecture Search]]
- [[concepts/algorithm-hardware-co-design|Algorithm-Hardware Co-Design]]

## Common confusions
- **Encoding non-numerical parameters:** Loop orders (such as `[C, R, X, K, Y, S]`) are discrete permutations with no natural numerical ordering. NAAS solves this using **importance-based encoding**: each loop dimension is assigned a continuous score, and sorting scores establishes the execution hierarchy, making discrete compiler schedules optimizable by gradient and evolutionary search algorithms.

## Appears in
- [[lessons/2026-11-01-hardware-aware-nas-and-once-for-all-networks]]: introduced as the ultimate realization of full-stack algorithm-compiler-silicon co-design.
