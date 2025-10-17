
---

# Flash Attention 2 (Triton Implementation)

An optimized implementation of **FlashAttention2** written from scratch using **Triton kernels**.  
This project demonstrates how memory-efficient attention can be achieved through tiling, streaming, and kernel fusion — enabling large-scale Transformer models to train and infer faster without exceeding GPU memory limits.

---

## Overview

**Flash Attention** rethinks the standard attention computation by avoiding the explicit construction of the attention matrix (`QKᵀ`).  
Instead, it computes softmax-normalized attention in blocks directly in GPU registers and SRAM — drastically improving performance and reducing memory overhead.

This repository reimplements **FlashAttention v2** using **Triton**, with an educational focus on:

- Kernel fusion  
- Memory-efficient tiling  
- Causal masking  
- GPU-level performance optimization  

---

## Motivation

Modern Transformer models are memory-bound:  
the quadratic cost of attention (O(n²)) limits sequence length and batch size.  
FlashAttention2 mitigates this by **computing attention in a streaming fashion**,  
avoiding redundant reads/writes and exploiting GPU hardware locality.

This project helps you:

- Understand how FlashAttention works at a kernel level  
- Learn Triton programming and GPU memory hierarchy  
- Experiment with custom attention variants and optimizations  

---

## Features

- **Forward + Backward** Triton kernels  
- **Memory-efficient** blockwise attention computation  
- **Causal masking** support  

---

## Architecture

The high-level data flow of the kernel:



Input: Q, K, V tensors
↓
Blockwise computation (tiling along sequence dimension)
↓
Partial dot-products (Q·Kᵀ per tile)
↓
Streaming softmax accumulation
↓
Multiply with V per tile
↓
Output aggregation



Key optimizations:

- **Tiling:** Operate on manageable chunks to stay within SRAM/registers  
- **Streaming softmax:** Avoids recomputation of normalization constants  
- **Fused compute:** Minimizes global memory reads/writes  

---

## Installation

```bash
git clone https://github.com/anirudh-qual/Flash-Attention.git
cd Flash-Attention
conda env create environment.yml
```
---

## Quick Start

Here’s how to test the FlashAttention kernel:

```python
import torch
from flash_attention import flash_attention  # adjust based on your file structure

B, S, D = 2, 128, 64  # batch, seq_len, hidden_dim
q = torch.randn(B, S, D, device="cuda")
k = torch.randn(B, S, D, device="cuda")
v = torch.randn(B, S, D, device="cuda")

out = flash_attention(q, k, v, causal=False)
print(out.shape)  # Expected: (B, S, D)
```

You can also benchmark against PyTorch’s scaled_dot_product_attention for validation.

---

## Benchmarks

| Sequence Length | Naive Attention (ms) | FlashAttention2 (ms) | Speedup |
| --------------- | -------------------- | -------------------- | ------- |
| 512             | 18.2                 | 6.1                  | 2.98×   |
| 1024            | 64.7                 | 19.4                 | 3.33×   |
| 2048            | 261.3                | 70.5                 | 3.70×   |

> Benchmarked on NVIDIA A40 GPU, PyTorch 2.4, Triton 2.1, FP16 precision.

---

## Future Work

* Mixed precision (FP8/BF16) variants
* Multi-head / grouped attention
* Integration with HuggingFace models
* Attention dropout support
* Additional kernel autotuning and profiling

---

## Acknowledgments

This implementation was **inspired by Umar Jamil’s FlashAttention tutorial and walkthrough**:
🎥 [YouTube — Building FlashAttention from Scratch](https://www.youtube.com/watch?v=zy8ChVd_oTM&t=16391s)

Special thanks to the open-source **FlashAttention**, **Triton**, and **PyTorch** communities for making high-performance deep learning accessible.

---

<p align="center">⭐ If you found this helpful, consider starring the repo!</p>
```

---
