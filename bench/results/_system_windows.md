# System Configuration - Windows Benchmark Machine

## Capture Info
- **Date**: 2026-01-05 01:14 PST (UTC-8)
- **Git Commit**: c22ee45

## Hardware

### GPU
- **Model**: NVIDIA GeForce RTX 4070 Laptop GPU
- **VRAM**: 8188 MiB (8 GB)
- **CUDA Cores**: 4608
- **Architecture**: Ada Lovelace (AD106)

### CPU
- Intel Core (see systeminfo.txt for full details)

### Memory
- System RAM: (see systeminfo.txt)

## Software

### NVIDIA Stack
- **Driver Version**: 591.44
- **CUDA Toolkit**: 12.8 (V12.8.93)
- **Build Date**: 2025-02-21

### Operating System
- **OS**: Windows 11
- **Build**: See systeminfo.txt

### Build Environment
- **Compiler**: NVCC 12.8.93
- **Build Flags**: -O3 -use_fast_math
- **Target**: Release

## Power Configuration
- **Mode**: Laptop (battery/plugged status not recorded)
- **Note**: For reproducible results, always benchmark while plugged in with "Best Performance" power mode

## Raw Data Files
- `systeminfo.txt` - Full Windows system info
- `cpu.txt` - CPU model
- `gpu.txt` - GPU identifier
- `gpu_details.txt` - GPU details (driver, memory)
- `nvcc.txt` - CUDA compiler version
