# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Benchmark and correctness test for a zero-copy IPC architecture targeting **Jetson Orin** VLA (Vision-Language-Action) robot inference. The core idea: POSIX shared memory + `cudaHostRegister` lets a server GPU process read observations written by a CPU client process with no serialization and no `cudaMemcpy` — because Jetson's unified LPDDR5 DRAM is physically shared between CPU and GPU.

## Setup

```bash
bash setup.sh          # create .venv (Python 3.10), install deps via uv
source .venv/bin/activate
```

Dependencies: `numpy==1.26.1`, `torch==2.6.0+cu126`, `cupy-cuda12x>=13,<14`, `posix-ipc`.

## Running

```bash
python test_shm.py                              # run all three tests
nsys profile --trace=cuda python test_shm.py   # with CUDA profiling
nsys stats report.nsys-rep                     # summarize a captured profile
```

## Architecture

Single file (`test_shm.py`) spawning two processes via `multiprocessing.Process` with `spawn` start method:

- **Server process** — creates the POSIX shm region, calls `cudaHostRegister` (via `ctypes` + `libcudart.so`) to pin pages and obtain a GPU device pointer, wraps it in `cp.cuda.UnownedMemory` to get a zero-allocation CuPy array. Also holds a `torch.Tensor` view for Test A only.
- **Client process** — opens the existing shm region, gets a CPU numpy array view via `np.frombuffer`. No CUDA involvement on the client side.

Synchronization uses four POSIX semaphores (`posix_ipc.Semaphore`): two for Tests A/B (one-shot), two for Test C's obs/act loop.

**Three tests:**
- **Test A** — client writes ascending floats via numpy; server reads back via GPU tensor and verifies.
- **Test B** — server writes `42.0` via CPU; client verifies.
- **Test C** — 100-iteration loop simulating obs→infer→act: client writes obs, server does `cp_array.sum()` (GPU kernel on shared memory), server writes act back, client reads.

## Key Implementation Details

**Why CuPy, not PyTorch for GPU access:** `torch.as_tensor(cpu_array).cuda()` always issues a `cudaMemcpy` even on Jetson. CuPy's `UnownedMemory` wraps the raw device pointer directly — no copy.

**Zero-copy is only zero-copy on Jetson** (`props.is_integrated == True`). On a discrete GPU (e.g. RTX 3090), `cudaHostGetDevicePointer` on mmap'd memory still causes PCIe stalls per cache miss. The report prints a warning for this case.

**POSIX resource cleanup:** The `server` process owns all shm/semaphore names and calls `unlink()` in its `finally` block. The `main` block also pre-cleans leftover names from crashed previous runs before spawning processes.

## Reference Codebase

The production VLA client-server architecture, model inference logic, and related utilities live in `~/Projects/Isaac-GR00T/Isaac-GR00T/`. Refer to that repo when working on anything related to the actual inference server, action chunking, TensorRT integration, or the full robot control pipeline.

Key files in the reference repo:
- `gr00t/policy/server_client.py` — `PolicyServer` (ZMQ REP) + `PolicyClient` + `MsgSerializer` (msgpack + numpy)
- `gr00t/policy/gr00t_policy.py` — `Gr00tPolicy` + `Gr00tSimPolicyWrapper` (flat-key format adapter)
- `gr00t/eval/sim/SimplerEnv/simpler_env.py` — `WidowXBridgeEnv`: image `(256, 256, 3)` uint8, 8 state floats
- `gr00t/configs/data/embodiment_configs.py` — per-embodiment observation/action shapes and delta_indices
- `scripts/eval/CLOSEDLOOP_EVAL.md` — end-to-end eval setup guide

See `ARCHITECTURE_NOTES.md` in this folder for a full breakdown of the current ZMQ transport, data shapes, and the planned shm migration work.

## Profiling Reports

The `.nsys-rep` / `.sqlite` files in the repo root are captured `nsys profile` outputs from four benchmark runs. Use `nsys stats <file>.nsys-rep` or open in Nsight Systems GUI to inspect CUDA API and kernel timelines.
