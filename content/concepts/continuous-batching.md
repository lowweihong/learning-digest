---
title: Continuous Batching
aliases: [iteration-level scheduling, dynamic batching, token-level batching, in-flight batching]
tags: [concept, llm, serving, systems, efficient-ai]
---

## Definition
An LLM inference serving schedule that manages GPU workloads at the token-iteration level rather than the request level, allowing requests that finish generation to immediately exit the active batch and newly arrived requests to join on the very next token decoding step.

## Why it's true
In traditional deep learning pipelines (like vision CNNs or BERT discriminative models), all inputs in a batch take identical execution time because forward passes are fixed. Batching multiple inputs together amortizes memory traffic and maximizes hardware utilization.

In generative large language models, this model breaks down completely due to **asymmetric generation lengths**:
1. **The Trapped Batch Problem:** Suppose a batch of 8 requests contains seven short requests (generating 15 tokens each) and one long request (generating 1,000 tokens).
2. **Idle Resource Waste:** In standard request-level batching, the entire batch must execute for 1,000 steps until the slowest request finishes. After step 15, seven out of eight GPU slots are generating wasted padding tokens, operating at $12.5\%$ hardware efficiency.
3. **Queue Latency Penalties:** Incoming requests submitted while the long request is running must wait in the queue, severely inflating Time-to-First-Token (TTFT).

**Continuous Batching (Iteration-Level Scheduling)** eliminates request-level batch boundaries:
- The serving engine reschedules GPU tensor slots after **every single token generation step**.
- The moment a request emits an `<EOS>` (End-of-Sequence) token, it is evicted from the active batch immediately, freeing its [[concepts/kv-cache|KV Cache]] blocks.
- A new request waiting in the admission queue is immediately inserted into the vacant slot. Its prompt is prefilled, and it seamlessly joins the ongoing batch on the very next step.

```
Static Batching (Wasted Idle Slots):
Req 1 (15 tokens):  [================] (IDLE PADDING FOR 985 STEPS ........................)
Req 2 (1000 tokens): [======================================================================]
New Req 3:          (WAITS IN QUEUE FOR 1000 STEPS BEFORE STARTING .........................)

Continuous Batching (100% Saturation):
Req 1 (15 tokens):  [================] ==> EXITS IMMEDIATELY!
Req 3 (Joins):                         [======================== New Request Starts! =======]
Req 2 (1000 tokens): [======================================================================]
```

Paired with dynamic memory allocators like [[concepts/pagedattention|PagedAttention]], continuous batching keeps GPU compute units saturated near $100\%$ capacity, multiplying serving throughput by **$2\times$ to $5\times$** while slashing average user latency.

## Minimal example
A server hosting 4 concurrent slots on an A100 GPU:
- **Static Batching:** Request A finishes in 20 tokens; Request B in 30 tokens; Request C in 40 tokens; Request D takes 200 tokens. The average GPU slot utilization over the 200 steps is only $\frac{20 + 30 + 40 + 200}{4 \times 200} = 36.2\%$.
- **Continuous Batching:** As Requests A, B, and C complete, waiting Requests E, F, G, and H immediately fill the open slots. The GPU maintains 4 active requests on every single step, raising hardware utilization to $100\%$ and tripling system throughput.

## Prerequisites
- None

## Common confusions
- **Continuous batching mixes prefill and decoding:** Yes. On a given iteration step, some requests may be performing prompt prefill (compute-bound) while others are decoding a single token (memory-bound). Modern chunked-prefill engines balance prefill and decode FLOPs to prevent prefill from stalling decode latency.
- **Continuous batching alters model outputs:** The technique modifies only the multi-tenant scheduling queue; individual tokens generated for each sequence are mathematically identical to those generated in isolation.

## Appears in
- [[lessons/2026-11-01-llm-quantization-and-deployment-techniques]]: introduced as the industry standard serving strategy used in vLLM, TensorRT-LLM, and TGI.
