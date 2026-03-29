# Architectural Visibility Bias: Cache Masking in Mobile PIM Evaluation

This repository contains the full implementation and experimental framework used to study how cache capacity and eviction policies mask the memory intensity of LLM inference kernels on mobile devices.

## Contents
- QEMU-TCG instrumentation plugin and L1/L2 cache models
- External LLC simulator  
- GEMV, RMSNorm, and RoPE kernels    

## Summary
The project demonstrates that shared LLCs in mobile SoCs suppress MPKI, LFMR, and AI, leading to incorrect conclusions about PIM suitability. The repository provides a reproducible pipeline for evaluating cache masking effects across multiple eviction policies and cache configurations.

## Students
Kirill Vyskubov  
Danil Vyskubov

## Supervised by
Prof. Oguz Ergin  
University of Sharjah — Senior Design Project II (2025/2026)
