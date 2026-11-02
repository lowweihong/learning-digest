---
title: Group Relative Policy Optimization
aliases: [GRPO, group relative reinforcement learning, DeepSeek-R1 RL, critic-free RL]
tags: [concept, llm, reinforcement-learning, reasoning, post-training]
---

## Definition
A reinforcement learning algorithm developed for reasoning models (notably **DeepSeekMath** and **DeepSeek-R1**) that eliminates the memory-heavy critic/value network in standard PPO by sampling a group of candidate completions for each prompt and normalizing rewards across the group to calculate baseline-free relative advantages.

## Why it's true
In standard Proximal Policy Optimization (**PPO**) for large language models, estimating the advantage $A(s, a) = R - V(s)$ requires training a separate **Critic (Value) Network**. In a 70B parameter model deployment:
- The critic network is comparable in size to the actor (policy) model ($70\text{B}$ parameters).
- Holding the actor, critic, reward model, and reference model simultaneously demands astronomical GPU VRAM, forcing severe constraints on batch sizes and sequence lengths.
- Training an accurate value function for long-horizon mathematical proofs and code generation is notoriously difficult and introduces massive training variance.

**Group Relative Policy Optimization (GRPO)** resolves this by replacing the learned critic network with empirical group statistics:
1. **Group Sampling:** For each question or prompt $q$, the policy $\pi_\theta$ samples a group of $G$ distinct candidate outputs: $\mathcal{O} = \{o_1, o_2, \dots, o_G\}$ (typically $G = 8$ to $16$).
2. **Rule-Based or Verifiable Rewards:** Each generated completion receives a scalar reward score $r_i$ based on verifiable correctness (e.g., automated unit test execution for coding, exact mathematical answer matching, or format adherence).
3. **Group-Normalized Advantage:** Instead of querying a critic network, the advantage for each candidate $o_i$ is computed by standardizing its reward against the mean and standard deviation of its peer group:
   $$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$
4. **Policy Gradient with KL Penalty:** The model is optimized using a clipped surrogate objective that increases the likelihood of tokens from above-average completions ($A_i > 0$) and penalizes tokens from below-average completions ($A_i < 0$), with an explicit KL divergence penalty against a frozen reference model $\pi_{\text{ref}}$:
   $$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left[ \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)} A_i - \beta D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \right]$$

By dispensing with the critic model, GRPO cuts training memory consumption nearly in half and eliminates value-estimation bias, enabling massive rollouts that cultivate emergent reasoning behaviors (like internal chain-of-thought self-reflection and error backtracking).

## Minimal example
Prompt: *"Compute 24 using the numbers 5, 5, 5, 1 and basic arithmetic operations."*
Sample $G = 4$ candidate rollouts:
- **Output 1:** $(5 - 1/5) \times 5 = (24/5) \times 5 = 24$ $\to r_1 = 1.0$ (Correct!)
- **Output 2:** $5 \times 5 - 1^5 = 24$ (Invalid: uses exponentiation) $\to r_2 = 0.0$
- **Output 3:** $(5 - (1/5)) \times 5 = 24$ $\to r_3 = 1.0$ (Correct!)
- **Output 4:** $5 + 5 + 5 + 1 = 16$ $\to r_4 = 0.0$ (Wrong result)

Group statistics: $\text{mean}(r) = 0.5$, $\text{std}(r) = 0.5$.
- Normalized advantages:
  $$A_1 = +1.0, \quad A_2 = -1.0, \quad A_3 = +1.0, \quad A_4 = -1.0$$
The policy parameters update to reinforce the exact valid arithmetic sequences of Outputs 1 and 3 without ever allocating memory for a critic network.

## Prerequisites
- None

## Common confusions
- **GRPO requires human raters:** GRPO is designed primarily for **verifiable domain reasoning** (math, logic, competitive programming) where rewards are computed deterministically via automated interpreters, compiler test harnesses, and regex parsers, requiring zero human annotation during the RL loop.
- **Rollout generation is cheap:** The rollout phase (sampling $G$ long responses token-by-token) consumes over $85\text{–}90\%$ of total training clock time. Systems often pair GRPO with speculative decoding (e.g., TOT) to keep GPU cores saturated during rollout generation.

## Appears in
- [[lessons/2026-11-01-llm-post-training-and-alignment]]: introduced as the core reinforcement learning algorithm powering frontier reasoning models like DeepSeek-R1.
