# Awesome Video Attention

A curated list of recent papers on **efficient video attention** for video diffusion models, including **sparsification**, **quantization**, and **caching**, etc.

> 📌 Sorted in **reverse chronological order** by arXiv submission date.

---

## Papers

- **[Sparse-vDiT: Unleashing the Power of Sparse Attention to Accelerate Video Diffusion Transformers](https://arxiv.org/abs/2506.03065)** (Jun 2025)  
This paper provides a detailed analysis of attention maps in Video Diffusion Transformers and identifies three recurring sparsity patterns: diagonal, multi-diagonal, and vertical-stripe structures. It achieves 2.09× theoretical FLOP reduction and 1.76× inference speedup on CogVideoX while maintaining visual fidelity.

- **[Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column‑Sparse Deltas](https://arxiv.org/abs/2506.03275)** (Jun 2025)  
  Exploits step-to-step activation redundancy in DiTs via dynamic sparsity and voxel-based token reordering. Implements efficient column-sparse GPU kernels and overlapping strategies to hide latency. Achieves up to 3.72× speedup (HunyuanVideo) without retraining or quality loss.

- **[Astraea: A GPU‑Oriented Token‑wise Acceleration Framework for Video Diffusion Transformers](https://arxiv.org/abs/2506.05096)** (Jun 2025)  
  Proposes an automatic framework combining lightweight token selection and GPU-parallel sparse attention. Uses evolutionary search to optimize token budgets across timesteps. Achieves up to 2.4× speedup on 1 GPU and 13.2× on 8 GPUs with <0.5% quality drop on VBench.

- **[PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models](https://arxiv.org/abs/2506.16054)** (Jun 2025)  
  Proposes a token reordering method that transforms irregular attention into hardware-friendly block-wise patterns, simplifying both sparsification and quantization. Achieves lossless visual generation with INT8/INT4 at ~20–30% density, yielding 1.9×–2.7× latency speedup.

- **[VMoBA: Mixture‑of‑Block Attention for Video Diffusion Models](https://arxiv.org/abs/2506.23858)** (Jun 2025)  
  Proposes a sparse mixture-of-block attention mechanism that partitions video tokens into 1D, 2D, and 3D blocks to exploit spatio-temporal locality, achieving ≈2.9× lower FLOPs and 1.35× faster training/inference in long video generation.

- **[Radial Attention: Sparse Attention with Energy Decay for Long Video Generation](https://www.arxiv.org/abs/2506.19852)** (Jun 2025)  
  Introduces a static $O(n\log n)$ attention mask inspired by spatiotemporal energy decay, enabling ~4× longer videos with up to 1.9× speedup over dense attention in pretrained video diffusion models.

- **[FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Fast Video Diffusion](https://arxiv.org/abs/2506.04648)** (Jun 2025)  
  Develops a training-aware co-design of FP8 quantization and structured sparsity for 3D attention, yielding 7.09× attention speedup (≈4.96× end-to-end) at 720p with negligible quality loss.

- **\[MICCAI 2025\][FEAT: Full-Dimensional Efficient Attention Transformer for Medical Video Generation](https://arxiv.org/abs/2506.04956)** (Jun 2025)<br>
Sequential spatial-temporal-channel attention with linear complexity; FEAT-S matches Endora with only 23% parameters.

- **[Interspatial Attention for Efficient 4D Human Video Generation](https://arxiv.org/abs/2505.15800)** (May 2025)<br>
Introduces interspatial attention (ISA) mechanism for diffusion transformer-based video generation models, using relative positional encodings tailored for human videos. Achieves state-of-the-art 4D human video synthesis with motion consistency and identity preservation.

- **[MAGI-1: Autoregressive Video Generation at Scale](https://arxiv.org/abs/2505.13211)** (May 2025)
`MagiAttention` is a distributed attention mechanism, or context-parallel (CP) strategy, which aims to support a wide variety of attention mask types with **kernel-level flexibility**, while achieving **linear scalability** with respect to context-parallel (CP) size across a broad range of scenarios, particularly suitable for training tasks involving <u><em>ultra-long, heterogeneous mask</em></u> training like video-generation for Magi-1.

- **[GRAT: Grouping First, Attending Smartly – Training-Free Acceleration for Diffusion Transformers](https://arxiv.org/abs/2505.14687)** (May 2025)  
  Proposes GRAT, a training-free strategy that partitions tokens into GPU-friendly groups and restricts attention to structured regions. Delivers up to 35.8× speedup in 8192×8192 generation on A100, while preserving quality on pretrained Flux and HunyuanVideo.

- **[Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation](https://arxiv.org/abs/2505.18875))** (May 2025)  
  Proposes a training-free method that reorders tokens based on semantic clustering to form dense blocks. Achieves up to 2.3× speedup with minimal quality drop.

- **[VSA: Efficient Video Diffusion via Routing Sparse Attention](https://arxiv.org/abs/2505.13389)** (May 2025)  
Proposes a trainable sparse attention mechanism that routes attention to important tokens in video diffusion models, reducing training FLOPs by 2.5× and cutting inference latency from 31s to 18s without degrading generation quality.

- **[VORTA: Efficient Video Diffusion via Routing Sparse Attention](https://arxiv.org/abs/2505.18809)** (May 2025)  
  Introduces a routing-based framework to replace full 3D attention with specialized sparse patterns during sampling. Delivers 1.76× end-to-end speedup (14.4× with distillation) without quality degradation.

- **[FastCAR: Cache Attentive Replay for Fast Auto-regressive Video Generation on the Edge](https://arxiv.org/abs/2505.14709)** (May 2025)  
  Exploits temporal redundancy by caching MLP outputs between frames to skip redundant decoding steps. Achieves >2.1× faster decoding and better energy efficiency for edge devices.

- **[VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)** (May 2025)<br>
 Two-stage coarse-to-fine sparse kernel yielding 6× attention speed-up and 2.53× training FLOP reduction on Wan-2.1 with no loss in diffusion loss.

- **[SageAttention3: Microscaling FP4 Attention and 8-Bit Training](https://arxiv.org/abs/2505.11594)** (May 2025)  
  Leverages FP4 Tensor Cores on Blackwell GPUs to reach 1038 TOPS attention throughput and explores 8-bit attention training with promising results.

- **[Analysis of Attention in Video Diffusion Transformers](https://arxiv.org/abs/2504.10317)** (Apr 2025)  
  Provides an in-depth study of attention in VDiTs, identifying three key attention properties—**Structure**, **Sparsity**, and **Sinks**. Shows that attention patterns are prompt-agnostic, sparsity methods aren’t universally effective, and attention sinks differ from language models. Suggests future directions to improve the efficiency-quality tradeoff :contentReference[oaicite:1]{index=1}.

- **[Generalized Neighborhood Attention: Multi-dimensional Sparse Attention at the Speed of Light](https://arxiv.org/abs/2504.16922)** (Apr 2025)  
  Introduces a unified framework (GNA) for local sparse attention patterns—sliding window, strided, and blocked—and provides an analytical performance simulator. Implements GNA on NVIDIA Blackwell FMHA kernels, achieving up to 1.3 PFLOPs/s and 28%–46% end-to-end speedup on Cosmos-7B, FLUX, and HunyuanVideo without fine-tuning.


- **\[ICML 25\][XAttention: Block Sparse Attention with Antidiagonal Scoring](https://arxiv.org/abs/2503.16428)** (Mar 2024)
Proposes a sparse attention method using antidiagonal scoring for efficient block pruning, achieving up to 13.5× speedup with minimal accuracy loss on long-context language and video benchmarks.

- **[Training-free and Adaptive Sparse Attention for Efficient Long Video Generation](https://arxiv.org/abs/2502.21079)** (Feb 2025)  
  AdaSpa uses blockified adaptive sparse attention with online cached search to reduce PFLOPs in long video generation while maintaining fidelity.

- **\[ICML 25\][SpargeAttention: Accurate Sparse Attention Accelerating Any Model Inference](https://arxiv.org/abs/2502.18137)** (Feb 2025)  
  Introduces a training-free two-stage filter for fast sparse attention inference. Achieves high speedup with no quality loss across LLMs, image, and video models.

- **[DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training](https://arxiv.org/abs/2502.07590)** (Feb 2025)  
  Leverages dynamic attention sparsity and hybrid parallelism to achieve 3.02× training throughput on large-scale VDiT models.

- **\[ICML 25\][Fast Video Generation with Sliding Tile Attention](https://arxiv.org/abs/2502.04507)** (Feb 2025)  
  Restricts attention to a sliding 3D window, accelerating attention by 2.8–17× over FlashAttention-2 and reducing end-to-end latency by 27% without quality loss.

- **\[ICML 25\][Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity](https://arxiv.org/abs/2502.01776)** (Feb 2025)  
  Classifies heads as spatial vs. temporal and skips irrelevant computations. Yields ≈2.3× end-to-end speedups on modern video diffusion models.

- **\[ICML 25\][SageAttention2: Efficient Attention with INT4 Quantization](https://arxiv.org/abs/2411.10958)** (Nov 2024)  
  Combines INT4 $QK^\top$ and FP8 $PV$ with outlier smoothing to reach 3× higher throughput than FlashAttention2 while retaining high accuracy.

- **\[ICLR 24\][SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration](https://arxiv.org/abs/2410.02367)** (Oct 2024)  
  Pioneers 8-bit attention using INT8+FP16 strategy with smoothing. Achieves 2.1×–2.7× speedups over baselines with negligible accuracy drop.

- **\[WACV 2025\][Efficient Video Object Segmentation via Modulated Cross-Attention Memory](https://arxiv.org/abs/2403.17937)** (Mar 2024)<br>
Long-term modulated cross-attention memory cuts GPU memory by 87% and lifts speed to 37 FPS on LVOS with 7.6× acceleration.

---

## Contributing

If you find your paper related to attention in video generation, feel free to open a pull request!

---

## License

This project is under the MIT License.
