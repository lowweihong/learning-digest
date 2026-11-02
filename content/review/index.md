---
title: Review questions
---

Questions from every lesson, newest first. Answers are folded, so try each question before you open it.

<!-- The daily run inserts new sections below this line: "## YYYY-MM-DD — [[lessons/<stem>|<title>]]" followed by that lesson's question callouts. -->

## 2026-11-01 — [[lessons/2026-11-01-automated-pruning-and-hardware-sparsity|Automated Pruning and Hardware Sparsity: From Reinforcement Learning to Silicon Speedup]]

> [!question]- Q1 (recall): What is the fundamental difference between static weight sparsity and dynamic activation sparsity, and why are their arithmetic savings multiplicative?
> Static weight sparsity is fixed at compile time because the zeroed weights remain constant across all user queries, allowing compilers to pre-plan storage layouts and memory access schedules. Dynamic activation sparsity (such as zeros produced by ReLU) varies on the fly for every single input query, requiring hardware to detect and skip zeros at runtime. Because a multiplication is bypassed if *either* the weight is zero or the activation is zero, a network with $90\%$ zero weights ($10\times$ reduction) and $70\%$ zero activations ($3.3\times$ reduction) achieves $10 \times 3.3 \approx 33\times$ total arithmetic reduction. Linked note: [[concepts/static-vs-dynamic-sparsity|Static vs Dynamic Sparsity]].

> [!question]- Q2 (explain-why): Why did Song Han's AMC reinforcement learning agent automatically choose to prune 1x1 convolutions gently while pruning 3x3 convolutions aggressively?
> A $1 \times 1$ pointwise convolution contains only a single parameter per channel and serves the critical function of cross-channel information mixing; it contains virtually zero spatial redundancy, making it hyper-sensitive to parameter removal. In contrast, a $3 \times 3$ convolution has 9 parameters per channel spanning a spatial neighborhood, containing high spatial redundancy that surviving weights can easily compensate for during retraining. The RL agent discovered this structural principle automatically by observing validation rewards. Linked note: [[concepts/automl-for-model-compression|AutoML for Model Compression]].

> [!question]- Q3 (apply): An engineer wants to use NetAdapt to reduce a vision model's latency on an edge NPU. Why should they construct an on-device latency lookup table rather than using FLOPs as the optimization objective?
> Theoretical FLOP counts correlate poorly with true execution latency on modern edge silicon. Hardware speed is governed by memory access bandwidth, cache hit rates, kernel launch overheads, and vector register alignment. A layer modification that cuts FLOPs by $30\%$ might yield $0\%$ speedup if memory bandwidth remains saturated. An on-device lookup table captures real physical execution times across tensor shapes directly on the target processor, ensuring optimization steps deliver guaranteed wall-clock latency drops. See [[concepts/netadapt|NetAdapt]].

> [!question]- Q4 (transfer): Why does standard 3D convolution cause runaway computational explosion when applied to sparse LiDAR point clouds in autonomous vehicles, and how does submanifold sparse convolution prevent it?
> Standard 3D convolutions slide dense kernels across 3D voxels. Even if $99\%$ of the input space is empty air, every $3 \times 3 \times 3$ kernel dilates non-zero activations to neighboring empty voxels, rapidly turning a sparse point cloud into a dense 3D tensor within a few layers. Submanifold sparse convolution solves this by enforcing that an output feature is computed *if and only if* the center coordinate was already occupied in the input, keeping the spatial sparsity pattern constant across all network depths. See [[concepts/sparse-convolution|Sparse Convolution]].

## 2026-11-01 — [[lessons/2026-11-01-pruning-and-sparsity-fundamentals|Pruning and Sparsity: Cutting Redundant Weights Without Losing Accuracy]]

> [!question]- Q1 (recall): Why does fine-grained (unstructured) pruning fail to provide speedup on standard GPUs despite removing 80% of parameters?
> Standard GPUs are SIMD/SIMT architectures designed around dense matrix multiplication (GEMM). Irregular, scattered zeros cause severe thread divergence and uncoalesced memory reads. Without specialized sparse hardware engines, standard GPU execution kernels spend as much time processing sparse indexing overhead as computing arithmetic. Linked note: [[concepts/fine-grained-pruning|Fine-Grained Pruning]].

> [!question]- Q2 (explain-why): Why does iterative pruning achieve significantly higher compression ratios than one-shot pruning?
> One-shot pruning eliminates all small parameters simultaneously, inflicting a massive shock that pushes the model far outside its low-loss basin and causing permanent accuracy collapse. Iterative pruning removes weights in gentle increments ($15\%$–$20\%$) and fine-tunes between rounds, giving surviving non-zero weights time to readjust their numerical values via backpropagation to heal the model before the next cut. Linked note: [[concepts/iterative-pruning|Iterative Pruning]].

> [!question]- Q3 (apply): You are deploying an image classification model to a fleet of off-the-shelf mobile devices running standard CPU inference engines. Should you use fine-grained pruning or structured channel pruning, and what trade-off must you accept?
> You should choose structured channel pruning. Standard mobile CPU runtime engines cannot accelerate arbitrary unstructured zeros. Structured pruning physically removes entire channels, producing a regular, smaller dense matrix that runs faster immediately on stock BLAS kernels. The trade-off is that structured pruning has a much lower maximum compression ratio: accuracy typically begins degrading past $30\%$–$50\%$ channel removal, whereas fine-grained pruning could have removed up to $80\%$–$90\%$ of raw parameters on disk. Linked note: [[concepts/structured-pruning|Structured Pruning]].

> [!question]- Q4 (transfer): In modern Transformer architectures, the multi-head self-attention block computes $A = \text{softmax}(QK^T / \sqrt{d_k})V$, which contains no static weights. How can the principles of sparsity still be applied to accelerate this calculation?
> Although the attention matrix $A$ contains no static weights, it exhibits dynamic activation sparsity: in long contexts, each token attends strongly to only a tiny fraction of other tokens, while the vast majority of attention weights approach zero after softmax. By identifying and skipping uninformative token pairs dynamically (sparse attention), hardware can avoid computing the full $N \times N$ key-value multiplications, mirroring the memory and compute savings of weight pruning in activation space. Linked note: [[concepts/n-m-sparsity|N:M Sparsity]].

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
