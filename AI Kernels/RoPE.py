#!/usr/bin/env python3
"""
rope_prefill_numpy.py — Rotary Positional Embedding (RoPE) Kernel

This script isolates the RoPE mathematical transformation used in modern LLMs.
It applies a 2D rotation to adjacent pairs in the hidden dimension.
"""

import sys
import time
import numpy as np

# -----------------------------------------------------------------------------
# Default parameters (Defaults to 16MB Working Set at FP32)
# -----------------------------------------------------------------------------
S = 1024  # Sequence Length
D = 1024  # Hidden Dimension

# Parse command-line arguments
args = sys.argv[1:]
if '--s' in args:
    S = int(args[args.index('--s') + 1])
if '--d' in args:
    D = int(args[args.index('--d') + 1])

# Ensure D is even (RoPE operates on pairs of coordinates)
if D % 2 != 0:
    print("Error: Hidden dimension (D) must be an even number.", file=sys.stderr)
    sys.exit(1)

# -----------------------------------------------------------------------------
# Memory Profiling & Setup
# -----------------------------------------------------------------------------
# FP32 uses 4 bytes per element
bytes_per_element = 4
matrix_size_mb = (S * D * bytes_per_element) / (1024 * 1024)

# We use 4 matrices: X (input), cos_cache, sin_cache, and Y (output)
total_wss_mb = matrix_size_mb * 4

print(f"[rope_kernel] Sequence (S)={S} | Dimension (D)={D} | Precision=FP32", file=sys.stderr)
print(f"[rope_kernel] Matrix Size: {matrix_size_mb:.2f} MB each", file=sys.stderr)
print(f"[rope_kernel] Total Working Set Size: {total_wss_mb:.2f} MB", file=sys.stderr)

# Initialize tensors (simulating the pre-computed caches and activation data)
X   = np.random.randn(S, D).astype(np.float32)
cos = np.random.randn(S, D).astype(np.float32)
sin = np.random.randn(S, D).astype(np.float32)
Y   = np.zeros((S, D), dtype=np.float32)

print(f"[rope_kernel] Tensors allocated. Starting RoPE pass...", file=sys.stderr)

# -----------------------------------------------------------------------------
# Target Kernel (ROI)
# -----------------------------------------------------------------------------
input("Press Enter to continue...\n")
t_start = time.time()


    
x0 = X[:, 0::2]  # Even indices
x1 = X[:, 1::2]  # Odd indices
    
c0 = cos[:, 0::2]
c1 = cos[:, 1::2]
    
s0 = sin[:, 0::2]
s1 = sin[:, 1::2]
    
# Apply the 2D rotation matrix math
# Y_even = x0 * cos - x1 * sin
# Y_odd  = x0 * sin + x1 * cos
Y[:, 0::2] = (x0 * c0) - (x1 * s0)
Y[:, 1::2] = (x0 * s1) + (x1 * c1)

t_end = time.time()
total_time = t_end - t_start

# -----------------------------------------------------------------------------
# Performance Metrics
# -----------------------------------------------------------------------------
# 4 multiplications and 2 additions per pair (D/2 pairs) = 6 ops per D elements
flops_per_iter = S * (D // 2) * 6
#total_gflops = (flops_per_iter * ITERATIONS) / total_time / 1e9

# Data moved = (Read X + Read Cos + Read Sin + Write Y) * ITERATIONS
data_moved_mb = total_wss_mb

print(f"[rope_kernel] Kernel done.", file=sys.stderr)
print(f"[rope_kernel] Total time : {total_time:.4f}s", file=sys.stderr)
#print(f"[rope_kernel] Performance: {total_gflops:.3f} GFLOPs/s", file=sys.stderr)
print(f"[rope_kernel] Data moved : ~{data_moved_mb:.1f} MB", file=sys.stderr)
