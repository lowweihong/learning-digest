---
title: Review questions
---

Questions from every lesson, newest first. Answers are folded, so try each question before you open it.

<!-- The daily run inserts new sections below this line: "## YYYY-MM-DD — [[lessons/<stem>|<title>]]" followed by that lesson's question callouts. -->

## 2026-11-01 — [[lessons/2026-11-01-basics-of-neural-networks-and-efficiency-metrics|Basics of Neural Networks and Efficiency Metrics]]

> [!question]- Q1 (recall): If a convolutional neural network requires 500 million Multiply-Accumulate operations (500 MMACs) to process one frame, how many FLOPs does it perform?
> Exactly 1 Billion FLOPs (1 GFLOP). Every MAC operation combines one multiplication and one addition ($a \leftarrow a + b \times c$), which equals 2 floating-point operations ($500 \times 10^6 \times 2 = 1.0 \times 10^9$). Linked note: [[concepts/macs-vs-flops|MACs vs FLOPs]].

> [!question]- Q2 (explain-why): Why does batching multiple inputs together improve throughput on a server GPU, but worsen individual latency for an interactive user?
> Batching groups multiple inputs into larger matrix multiplications, which keeps parallel GPU tensor cores saturated and amortizes memory loading overhead across multiple samples (producing more total outputs per second). However, individual requests must wait in a queue for the batch to assemble and take longer to compute together, increasing the elapsed time from request arrival to response. Linked note: [[concepts/latency-vs-throughput|Latency vs Throughput]].

> [!question]- Q3 (apply): A standard $3 \times 3$ convolution takes an input with 128 channels and produces 128 output channels. Calculate how many arithmetic operations are saved per pixel if this layer is replaced with a depthwise separable convolution.
> Standard convolution requires $3 \times 3 \times 128 \times 128 = 147,456$ operations per pixel. Depthwise separable convolution requires a depthwise step ($3 \times 3 \times 128 = 1,152$) plus a pointwise $1 \times 1$ step ($1 \times 1 \times 128 \times 128 = 16,384$), totaling $17,536$ operations. This saves $147,456 - 17,536 = 129,920$ operations per pixel—an $8.4\times$ reduction in compute. Linked note: [[concepts/depthwise-separable-convolution|Depthwise Separable Convolution]].

> [!question]- Q4 (transfer): An embedded engineer compresses a vision model from 80 MB down to 15 MB to fit inside a drone's onboard RAM. When deployed, the drone's processor immediately triggers an out-of-memory crash during the first layer. Why did this occur, and where in the network does memory pressure peak?
> Model size only measures static weight memory on disk. During inference, the processor must also allocate dynamic activation memory for intermediate feature maps. In the earliest layers, image resolution ($H \times W$) is at its highest, meaning activation tensors consume far more memory than the small convolutional kernels. The model crashed because peak activation memory exceeded available RAM. Linked note: [[concepts/activation-memory-vs-weight-memory|Activation Memory vs Weight Memory]].

## 2026-11-01 — [[lessons/2026-11-01-intro-to-tinyml-and-efficient-ai|Introduction to TinyML and Efficient AI Computing]]

> [!question]- Q1 (recall): Why does moving data between memory (DRAM) and computing cores consume dramatically more energy than performing arithmetic operations?
> Wires connecting external memory chips to the processor are long, capacitive, and off-chip. Shuttling a 32-bit number across those wires consumes up to 200× more energy (~1,000 picojoules) than executing an arithmetic multiplication or addition (~5 picojoules) inside local on-chip registers. Linked note: [[concepts/the-memory-compute-gap|The Memory-Compute Gap]].

> [!question]- Q2 (explain-why): Why has Algorithm-Hardware Co-Design become mandatory for AI performance gains, rather than relying solely on chip manufacturing advances?
> Traditional chip scaling rules (Moore's Law and Dennard scaling) have plateaued: clock frequencies and single-threaded performance have hit thermal ceilings. Modern speedups can no longer come from faster clock cycles alone; they require designing neural network architectures and memory access patterns specifically tailored to silicon cache lines, tensor cores, and SIMD registers. Linked note: [[concepts/algorithm-hardware-co-design|Algorithm-Hardware Co-Design]].

> [!question]- Q3 (apply): An edge developer wants to deploy a 14 GB vision-language model onto a portable device with only 4 GB of available RAM. What compression technique should they apply, and by what factor must memory footprint shrink to fit?
> They need to reduce the model's footprint by at least $3.5\times$ (from 14 GB to under 4 GB). By applying 4-bit weight quantization (such as AWQ), each 16-bit weight is compressed from 2 bytes down to 0.5 bytes, yielding an exact $4\times$ memory reduction and bringing the model down to ~3.5 GB, comfortably fitting inside the 4 GB limit. Linked note: [[concepts/tinyml|TinyML]].

> [!question]- Q4 (transfer): In mobile photography, smartphone cameras use on-device machine learning to enhance photos in real time. Why is on-device training significantly more difficult to implement on a smartphone or drone than on-device inference?
> Inference only requires passing activations forward through the network and discarding intermediate results after each layer finishes. Training requires executing a backward pass (backpropagation), which demands storing all intermediate layer activations in memory simultaneously to compute weight gradients, requiring roughly 3× more compute and orders of magnitude more memory. Linked note: [[concepts/on-device-training|On-Device Training]].

## 2026-11-01 — [[lessons/2026-11-01-quantization-k-means-and-linear|Neural Network Quantization: From Physical Bits to Integer Arithmetic]]

> [!question]- Q1 (recall): Why does K-means weight sharing reduce memory traffic but fail to speed up arithmetic on low-power chips without an FPU?
> K-means stores weights as compact integer bucket IDs, saving memory transfer bandwidth. However, during computation, each ID must be looked up in a table to fetch the original 32-bit decimal centroid, meaning the processor still has to perform full floating-point arithmetic. Linked note: [[concepts/k-means-quantization|K-Means Quantization]].

> [!question]- Q2 (explain-why): Why must the zero-point $Z$ in linear quantization be an exact integer rather than a decimal number?
> The zero-point lives in the integer domain and is directly added to or subtracted from integer values inside hardware registers. If $Z$ were a decimal, the processor would be forced to convert everything back into floating-point math, defeating the goal of running pure integer arithmetic. Linked note: [[concepts/zero-point|Zero-Point]].

> [!question]- Q3 (apply): A layer has decimal values between $R_{\min} = -2.0$ and $R_{\max} = 6.0$. Calculate the scale factor $S$ and zero-point $Z$ for standard signed 8-bit integers ($Q_{\min} = -128, Q_{\max} = 127$).
> The scale factor is $S = \frac{R_{\max} - R_{\min}}{Q_{\max} - Q_{\min}} = \frac{6.0 - (-2.0)}{127 - (-128)} = \frac{8.0}{255} \approx 0.03137$. The zero-point is $Z = \text{round}\left(Q_{\min} - \frac{R_{\min}}{S}\right) = \text{round}\left(-128 - \frac{-2.0}{0.03137}\right) = \text{round}(-128 + 63.75) = \text{round}(-64.25) = -64$. Linked note: [[concepts/linear-quantization|Linear Quantization]].

> [!question]- Q4 (transfer): In voice compression, Vector Quantization clusters sound waves into codebook buckets. How does neural network K-means fine-tuning differ from standard audio clustering, and what would happen if you re-clustered weights with standard K-means after every training batch?
> Standard clustering updates centroids to minimize geometric distance to data points. Neural network fine-tuning updates centroids using task gradients from backpropagation ($\Delta c_k = -\eta \sum \frac{\partial L}{\partial w}$) to minimize task loss. Re-clustering with standard K-means every batch would randomly shuffle weights into different buckets based on distance, destroying training momentum and ignoring whether predictions are actually accurate. Linked note: [[concepts/k-means-quantization|K-Means Quantization]].
