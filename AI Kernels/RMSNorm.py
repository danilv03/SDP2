#!/usr/bin/env python3
"""
rmsnorm_prefill_numpy.py — RMSNorm for LLM Prefill Phase

This script isolates a Root Mean Square Normalization layer.
During the prefill phase of an LLM, this operation must sweep across 
a massive 2D activation tensor (Sequence Length x Hidden Dimension).
"""

import sys
import time
import numpy as np

# -----------------------------------------------------------------------------
# Default parameters (Simulating a large prompt in a mobile LLM)
# -----------------------------------------------------------------------------
S     = 2048    # Sequence Length (number of tokens in the prompt)
D     = 4096    # Hidden Dimension size
DTYPE = np.float32

for i, arg in enumerate(sys.argv[1:]):
    if arg == '--s' and i+2 <= len(sys.argv): S = int(sys.argv[i+2])
    if arg == '--d' and i+2 <= len(sys.argv): D = int(sys.argv[i+2])

# -----------------------------------------------------------------------------
# Memory Profiling & Setup
# -----------------------------------------------------------------------------
bytes_per_element = np.dtype(DTYPE).itemsize

# The activation tensor (X) and the output tensor (Y) are both S x D
tensor_kb = (S * D * bytes_per_element) // 1024
# The static scaling weight (gamma) is just 1D (D elements)
gamma_kb  = (D * bytes_per_element) // 1024
total_kb  = (2 * tensor_kb) + gamma_kb

print(f"[rmsnorm_prefill] S={S} (tokens)  D={D} (dim)  dtype={np.dtype(DTYPE).name}", file=sys.stderr)
print(f"[rmsnorm_prefill] Input (X)={tensor_kb}KB  Output (Y)={tensor_kb}KB", file=sys.stderr)
print(f"[rmsnorm_prefill] Total Working Set={total_kb}KB", file=sys.stderr)

# -----------------------------------------------------------------------------
# Initialize tensors
# -----------------------------------------------------------------------------
# X represents the massive intermediate activation tensor
X = np.random.randn(S, D).astype(DTYPE)
# gamma represents the learned scaling weights for this layer
gamma = np.random.randn(D).astype(DTYPE)
epsilon = 1e-5

print(f"[rmsnorm_prefill] Tensors initialized. Starting RMSNorm pass...", file=sys.stderr)
input("Press Enter to continue...\n")
# -----------------------------------------------------------------------------
# Target Kernel (ROI)
# -----------------------------------------------------------------------------
t_start = time.time()

# 1. Calculate the mean of the squares along the hidden dimension (D)
# This forces the CPU to read the entire matrix X
variance = np.mean(X**2, axis=1, keepdims=True)

# 2. Calculate the inverse square root
inv_rms = 1.0 / np.sqrt(variance + epsilon)

# 3. Scale the original matrix X by the inverse RMS and the learned weights (gamma)
# This forces the CPU to read the entire matrix X a SECOND time
Y = X * inv_rms * gamma

t_end = time.time()
total_time = t_end - t_start

# -----------------------------------------------------------------------------
# Performance & Data Movement Metrics
# -----------------------------------------------------------------------------
# Math: square, sum, multiply, add, sqrt, div, mul, mul -> ~8 operations per element
flops = 8.0 * S * D
gflops = flops / total_time / 1e9

print(f"[rmsnorm_prefill] Kernel done.", file=sys.stderr)
print(f"[rmsnorm_prefill] Total time : {total_time:.4f}s", file=sys.stderr)
print(f"[rmsnorm_prefill] Performance: {gflops:.3f} GFLOPs/s", file=sys.stderr)
print(f"[rmsnorm_prefill] Data moved : ~{total_kb // 1024} MB", file=sys.stderr)
