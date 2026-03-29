#!/usr/bin/env python3
"""
gemv_decode_numpy.py — Matrix-Vector Multiplication for LLM Decode Phase

This script isolates the core operation of autoregressive token generation:
multiplying a massive weight matrix (W) by a small activation vector (x).
"""

import sys
import time
import numpy as np

# -----------------------------------------------------------------------------
# Default parameters (Configured for a realistic Mobile GenAI hidden layer)
# -----------------------------------------------------------------------------
N      = 2048   # Hidden dimension size (e.g., 4096x4096 weight matrix)
TSTEPS = 1     # Number of tokens to generate (timesteps)
DTYPE  = np.float32 # Typical inference precision (could also be float16)

# Parse arguments for adjustable working set
for i, arg in enumerate(sys.argv[1:]):
    if arg == '--n'      and i+2 <= len(sys.argv): N      = int(sys.argv[i+2])
    if arg == '--tsteps' and i+2 <= len(sys.argv): TSTEPS = int(sys.argv[i+2])

# -----------------------------------------------------------------------------
# Memory Profiling & Setup
# -----------------------------------------------------------------------------
bytes_per_element = np.dtype(DTYPE).itemsize

# The weight matrix is N x N
matrix_kb = (N * N * bytes_per_element) // 1024
# The input and output vectors are N x 1
vector_kb = (N * bytes_per_element) // 1024
total_kb  = matrix_kb + (2 * vector_kb)

print(f"[gemv_decode] N={N}  TSTEPS={TSTEPS}  dtype={np.dtype(DTYPE).name}", file=sys.stderr)
print(f"[gemv_decode] Weights (W)={matrix_kb}KB  Vectors (x, y)={vector_kb}KB each", file=sys.stderr)
print(f"[gemv_decode] Total Working Set={total_kb}KB", file=sys.stderr)

# Assess cache pressure (assuming typical mobile CPU caches)
if total_kb > 8192:
    print(f"[gemv_decode] Working set EXCEEDS typical mobile LLC ({total_kb}KB > 8192KB). Perfect for PIM.", file=sys.stderr)
elif total_kb > 1024:
    print(f"[gemv_decode] Working set stresses LLC ({total_kb}KB).", file=sys.stderr)
else:
    print(f"[gemv_decode] WARNING: working set fits in L2/LLC — increase N to measure memory bound.", file=sys.stderr)

# -----------------------------------------------------------------------------
# Initialize tensors
# -----------------------------------------------------------------------------
# W represents the static weights of a single linear layer
W = np.random.randn(N, N).astype(DTYPE)
# x represents the input token activations
x = np.random.randn(N).astype(DTYPE)

print("\nEverything for generation has been loaded. Please approve to continue to load the inputs and start the generation.")
input("Press Enter to continue...\n")

print(f"[gemv_decode] Tensors initialized. Starting decode loop...", file=sys.stderr)

# -----------------------------------------------------------------------------
# Target Kernel (ROI)
# -----------------------------------------------------------------------------
t_start = time.time()

for t in range(TSTEPS):
    # GEMV: Matrix-Vector Multiplication
    # Conceptually: y = W * x
    y = W @ x
    
    # Simulate feeding the output as the next input (autoregressive nature)
    # We add a tiny non-linearity/scaling to prevent overflow during profiling
    x = y * 0.01 

    if (t + 1) % 5 == 0 or t == 0:
        elapsed = time.time() - t_start
        print(f"[gemv_decode] Token {t+1:3d}/{TSTEPS} generated ({elapsed:.2f}s)", file=sys.stderr)

t_end = time.time()
total_time = t_end - t_start

# -----------------------------------------------------------------------------
# Performance & Data Movement Metrics
# -----------------------------------------------------------------------------
# FLOPs = 2 * N^2 per timestep (1 multiply, 1 add per matrix element)
flops_per_step = 2.0 * N * N
total_gflops = (flops_per_step * TSTEPS) / total_time / 1e9

# Data moved = Reading W, reading x, writing y per timestep
data_moved_mb = (total_kb * TSTEPS) / 1024

print(f"[gemv_decode] Kernel done.", file=sys.stderr)
print(f"[gemv_decode] Total time : {total_time:.3f}s", file=sys.stderr)
print(f"[gemv_decode] Performance: {total_gflops:.3f} GFLOPs/s", file=sys.stderr)
print(f"[gemv_decode] Data moved : ~{data_moved_mb:.1f} MB total", file=sys.stderr)
