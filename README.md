# VLA Inference Transport on Jetson Orin — Shared Memory Architecture

## 1. The Problem

A VLA (Vision-Language-Action) robot system naturally splits into two processes:

- **Client** — the robot controller. Runs at fixed frequency (e.g. 50 Hz), reads camera, sends actions to motors.
- **Server** — the inference engine. Loads a large VLA model, runs GPU inference, returns actions.

The naive implementation connects them with a socket or ZMQ:

```
Client                         Server
  │                              │
  │── serialize obs ──────────►  │
  │          (TCP/ZMQ socket)    │── model(obs) ──► GPU
  │                              │
  │◄── serialize act ──────────  │
```

This introduces unnecessary overhead **even on the same machine**:

- Serialization / deserialization of large tensors
- Syscall overhead for socket send/recv
- Memory copies: client RAM → kernel buffer → server RAM → GPU VRAM

For a full camera observation (e.g. `8×480×640` float32 = ~30 MB) at 50 Hz, this overhead is significant and adds latency to every inference call.

---

## 2. The Insight: Jetson Orin Unified Memory

On a standard workstation with a discrete GPU, memory is physically separated:

```
┌─────────────┐   PCIe (~16 GB/s)   ┌─────────────┐
│  CPU DRAM   │ ◄─────────────────► │  GPU VRAM   │
│  (DDR5)     │                     │  (GDDR6X)   │
└─────────────┘                     └─────────────┘
```

Every `tensor.cuda()` call copies data across this bus. For 30 MB at 16 GB/s, that is ~2 ms per transfer — and you need one in each direction per inference call.

On **Jetson Orin**, CPU and GPU share the same physical LPDDR5 DRAM chip:

```
┌──────────────────────────────────────┐
│         Shared LPDDR5 DRAM           │
│                                      │
│   CPU cores ──────── GPU cores       │
│         (same physical chip)         │
└──────────────────────────────────────┘
```

There is no PCIe bus. A pointer to "CPU memory" and "GPU memory" can refer to the same physical bytes. This means inter-process data sharing between the client and server can be **zero-copy end to end**.

### Benchmark: Kernel execution for 30 MB reduction

| Device | Kernel exec (avg) | Memory path |
|---|---|---|
| Jetson Orin | 123 µs | Local LPDDR5 (~204 GB/s) |
| RTX 3090 (pinned host mem) | 946 µs | PCIe (~16 GB/s) |

The 3090 is crippled not by compute but by memory transport. If data were in VRAM, a 3090 would be ~30x faster than Jetson. But with data coming from CPU RAM the PCIe bus is the bottleneck, making Jetson's unified memory architecture the right tool for this use case.

---

## 3. The Solution: POSIX shm + cudaHostRegister

Replace the socket with a shared memory region that both CPU and GPU can access directly, without copies.

### Mechanism

```
1. Allocate a POSIX shared memory region (OS primitive, cross-process by design)
2. mmap it in both client and server — each gets a CPU virtual address
3. Call cudaHostRegister() on the server side — pins the pages and tells the
   CUDA driver this memory is GPU-accessible
4. Call cudaHostGetDevicePointer() — returns a GPU-side virtual address
5. On Jetson: CPU VA and GPU VA map to the same physical DRAM pages
```

### What each process sees

```
POSIX shm (physical DRAM)
        │
        ├── Client mmap → cpu_ptr_client   (numpy array view)
        │
        └── Server mmap → cpu_ptr_server   (numpy array view)
                └── cudaHostRegister
                        └── cudaHostGetDevicePointer → gpu_ptr
                                └── CuPy UnownedMemory(gpu_ptr) → cp_array
```

All of `cpu_ptr_client`, `cpu_ptr_server`, and `gpu_ptr` point to the same physical bytes. No serialization. No copies in the hot loop.

### Why CuPy instead of PyTorch for GPU access

`torch.as_tensor(cpu_array).cuda()` always allocates a new GPU buffer and issues a `cudaMemcpy`, even on Jetson. PyTorch has no public API to wrap a raw device pointer.

CuPy's `UnownedMemory` wraps the raw `gpu_ptr` directly:

```python
mem    = cp.cuda.UnownedMemory(gpu_ptr.value, nbytes, owner=None)
memptr = cp.cuda.MemoryPointer(mem, 0)
arr    = cp.ndarray(shape, dtype=cp.float32, memptr=memptr)
```

This creates a GPU array view with zero allocation and zero copy. For the real inference server, TensorRT accepts raw device pointers via `context.set_tensor_address()` and writes output directly to the shared buffer — also zero copy.

### Synchronization

Process coordination uses POSIX semaphores — lightweight kernel primitives, ~1 µs overhead:

```python
# Client signals observation is ready
sem_obs.release()

# Server waits, runs inference, signals action is ready
sem_obs.acquire()
# ... inference ...
sem_act.release()

# Client reads action
sem_act.acquire()
```

No socket, no serialization, no polling. The semaphore is the only IPC mechanism needed.

---

## 4. Profiling Results

Tested on `SHAPE = (8, 480, 640)` float32 (~30 MB), 100 loop iterations.

### CUDA API (CPU side)

| Call | Orin avg | 3090 avg | notes |
|---|---|---|---|
| `cudaHostRegister` | 52 ms | 165 ms | one-time setup cost |
| `cudaLaunchKernel` | 17 µs | 44 µs | CPU kernel submission latency |
| `cudaMemcpyAsync` | 2.2 ms | 7.4 ms | explicit copy (Test A print only) |

### CUDA GPU Kernels (GPU side, 100 iterations of `cp_array.sum()`)

| Kernel | Orin avg | 3090 avg | ratio |
|---|---|---|---|
| `DeviceReduceKernel` | 123 µs | 946 µs | **7.7x** |

The 7.7x gap in kernel execution is entirely memory bandwidth: Jetson reads from local DRAM, 3090 fetches over PCIe per cache miss. No compute difference — the bottleneck is the bus.

### What nsys measures

- **CUDA API Summary** — CPU-side time: kernel submission, memcpy initiation
- **CUDA GPU Kernel Summary** — GPU-side time: actual execution including memory fetch stalls
- Kernel execution time on mapped pinned memory includes PCIe stall time per cache miss — to decompose further (compute vs memory stall), use `ncu` (Nsight Compute)

---

## 5. Current Limitations

### Atomic operations and memory ordering

The current implementation uses plain C integer fields in the shared header for coordination. On ARM64 (Jetson), aligned 32-bit loads/stores are hardware-atomic for single-producer-single-consumer (SPSC) patterns. This is safe for the current design but would break with multiple writers.

For production, wrap control fields in `std::atomic<int>` with `release`/`acquire` semantics via a small C extension, or use `multiprocessing.Value` with appropriate fencing.

### mmap page pinning

`cudaHostRegister` over an `mmap`'d region forces page pinning after the fact. Pages are not guaranteed to be contiguous or pre-faulted. For large buffers, prefer `cudaMallocHost` (which allocates pre-pinned memory) and share via `memfd_create` + file descriptor passing. This is more complex but avoids potential pinning failures on large allocations.

### Python GIL and GC jitter

The client is purely I/O-bound at ≤100 Hz, so the GIL is not a problem. However, Python's garbage collector can introduce 10–50 ms pauses in the control loop. Mitigate with:

```python
import gc
gc.disable()   # in the hot loop
```

For control rates above ~500 Hz or hard real-time requirements, rewrite the client in C++ with `SCHED_FIFO` scheduling and `mlockall`.

---

## 6. Future: Async Inference + Action Chunking

The current design is synchronous — the client blocks waiting for the server to finish each inference call. For a VLA model that takes 50–200 ms per inference, this freezes the robot during every call.

The solution is to decouple observation writing from action reading using three shared structures:

### Double-buffered observation

The client alternates between two obs buffers while the server always reads the most recently completed one. No contention, no blocking.

```
obs_buf[0] ◄── client writes (even ticks)
obs_buf[1] ◄── client writes (odd ticks)
               server reads whichever is latest (atomic index)
```

### Action chunk ring buffer

VLA models naturally produce chunks of actions (e.g. 10 steps). The server writes complete chunks to a ring buffer as they finish. The client consumes them one step at a time at control frequency.

```
┌──────┬──────┬──────┬──────┐
│chunk0│chunk1│chunk2│chunk3│  ← server writes (write_head)
└──────┴──────┴──────┴──────┘
                 ▲
           client reads one action per tick (read_head)
```

### Lock-free control header

A small struct at the start of the shared region holds atomic indices:

```c
typedef struct {
    atomic_int  obs_ready_idx;     // client publishes latest obs slot
    atomic_int  act_write_head;    // server advances after each chunk
    atomic_int  act_read_head;     // client advances after consuming chunk
    int         chunk_len;         // actions per inference (e.g. 10)
    int         action_dim;        // e.g. 7 for 7-DOF arm
} SharedHeader;
```

### Resulting timeline

```
t=0    client writes obs[0]
t=0    server starts infer(obs[0])
t=50ms server writes chunk[0] (10 actions) → ring[0]
t=50ms client starts executing chunk[0], action by action at 50 Hz
t=50ms server immediately starts infer(obs[1])  ← no idle time
t=100ms client finishes chunk[0], reads chunk[1] (already ready)
```

The robot never freezes. The server never idles. The ring buffer absorbs timing jitter between inference and control.

### Full memory layout

```
┌─────────────────────────────────────────────────────────────┐
│                   Shared LPDDR5 DRAM (Jetson)               │
│                                                             │
│  [ Header 64B ] [ obs_buf[0] ] [ obs_buf[1] ] [ ring × N ] │
│                                                             │
│  Client: CPU writes to obs_buf, CPU reads from ring         │
│  Server: GPU reads from obs_buf, GPU writes to ring via TRT │
│                                                             │
│  No serialize. No socket. No copies.                        │
└─────────────────────────────────────────────────────────────┘
```

### TensorRT integration

In the production server, `cp_array.sum()` is replaced by a TRT engine that reads obs and writes actions directly into the shared buffer:

```python
# point TRT I/O bindings at shared GPU pointers
context.set_tensor_address("obs",     obs_gpu_ptr)
context.set_tensor_address("actions", act_gpu_ptr)

# inference writes directly into the ring slot — zero copy
context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
arena.publish_chunk()
```

No intermediate allocations. No memcpy. TRT is compiled for SM 8.7 on Jetson so all kernel ops work correctly.

---

## 7. Summary

| Property | Socket/ZMQ | shm + cudaHostRegister (current) | + async chunks (future) |
|---|---|---|---|
| Serialization overhead | Yes | None | None |
| CPU→GPU copy | Yes | None on Jetson | None on Jetson |
| Robot freezes during inference | Yes | Yes | **Never** |
| Action chunking | No | No | Yes |
| Lock-free hot loop | No | Partial | Yes |
| TRT zero-copy output | No | No | **Yes** |
| Suitable for Jetson Orin | Poor | ✓ | ✓ |
| Suitable for discrete GPU | Reasonable | Poor (PCIe) | Poor (PCIe) |

---

## 8. Concrete Analysis Against GR00T VLA (OXE_WIDOWX)

The sections above use a synthetic `(8, 480, 640)` float32 = 30 MB tensor as a stand-in. The actual GR00T VLA payload for the WidowX robot (the target deployment) is substantially different. Here is what changes when we apply this architecture to the real system.

### Real payload sizes

The current GR00T server (`run_gr00t_server.py`) uses ZMQ REQ-REP with `MsgSerializer` (msgpack + per-array `np.save()` encoding). The client sends:

| Field | Shape | Dtype | Raw bytes |
|---|---|---|---|
| `video.image_0` (1 camera) | `(1, 1, 256, 256, 3)` | uint8 | **196,608 B** (~192 KB) |
| 8 state fields (x, y, z, roll, pitch, yaw, pad, gripper) | `(1, 1, 1)` each | float32 | **128 B** total |
| Language instruction | string | — | negligible |
| **Total observation** | | | **~193 KB** |

The server returns action keys `{x, y, z, roll, pitch, yaw, gripper}` each `(1, 8, 1)` float32 = **224 B** total. The response is tiny — all the cost is on the observation side.

This is **155× smaller** than the 30 MB benchmark tensor used in the tests above. What does that change?

### Transport overhead: measured numbers

Benchmarked on this machine via `test_zmq_baseline.py` (200 iterations, ZMQ TCP loopback, dummy server — no model, no GPU):

| Component | B=1 (real robot) | B=5 (sim eval) |
|---|---|---|
| Request payload | 193.6 KB | 962 KB |
| serialize (msgpack + np.save) | 0.30 ms | 1.29 ms |
| socket + kernel copy | 1.41 ms | 2.27 ms |
| deserialize (np.load + unpack) | 0.58 ms | 0.72 ms |
| **total round-trip (p50)** | **2.24 ms** | **4.26 ms** |
| **total round-trip (p99)** | **2.86 ms** | **4.94 ms** |

These are pure transport costs — the model inference (~100–200 ms PyTorch, ~20–50 ms TRT KV+FBC) adds on top. On Jetson the CPU serialization will be similar; the socket copy may be slightly slower.

With shm + `cudaHostRegister`, the transport cost drops to:
- Semaphore `release()` (POSIX futex): ~**1 µs**
- Semaphore `acquire()` (blocks until server signals): ~**0 µs** (already ready)
- **Measured total: ~2–5 µs** (from `test_shm.py` Test C loop)

Transport savings: **~2.2 ms (B=1) / ~4.3 ms (B=5) per call**, or roughly **400–2000× faster transport**.

### Does 2 ms matter?

That depends entirely on the control rate and where inference sits:

| Scenario | Inference time | Transport | Transport % | With shm |
|---|---|---|---|---|
| PyTorch BF16, real robot 10 Hz | ~100–200 ms | ~1–2 ms | ~1% | Minimal gain |
| TRT KV+FBC, real robot 30 Hz | ~20–50 ms | ~1–2 ms | ~4–10% | Modest gain |
| TRT KV+FBC, real robot 50 Hz target | ~20–50 ms | ~1–2 ms | ~4–10% | **Headroom matters** |
| TRT KV+FBC, real robot 50 Hz + action chunking | chunk overhead → ~5 ms slot | ~1–2 ms | **20–40%** | **Significant** |

The transport cost becomes proportionally large once TRT drives inference down toward 10–20 ms. It also compounds: the robot is blocked for `inference_time + transport_overhead` per cycle. On Jetson, where inference is already constrained, every saved millisecond extends the achievable control rate.

### The real bottleneck is not what the benchmark shows

The synthetic 30 MB test was chosen to dramatize the PCIe vs unified-memory gap on the GPU side. For the actual WidowX payload (192 KB uint8 image), the GPU kernel time drops accordingly:

- `cp_array.sum()` over 196 KB at 204 GB/s (Orin LPDDR5): ~**1 µs** (vs. 123 µs for 30 MB)
- The GPU computation on 192 KB is effectively instantaneous.

The dominant cost in the real system is not GPU memory bandwidth — it is **CPU-side serialization and socket overhead**. The benchmark results reported above (7.7× kernel speedup, Orin vs 3090) are real but only relevant if the observation is large (full-res multi-camera). For the current 256×256 single-camera WidowX setup, the Orin advantage is in eliminating the 1–2 ms serialization loop, not in kernel execution time.

### What the new architecture actually buys (current synchronous design)

1. **Eliminates serialization entirely.** The 1–2 ms CPU overhead per call (msgpack + np.save/load + socket copy) goes to zero. This is the main win at WidowX payload sizes.

2. **Eliminates the CPU→GPU copy on Jetson.** The server's CuPy array view (`cp.cuda.UnownedMemory`) wraps the shm region directly — no `cudaMemcpy`. This saves ~0.1–0.3 ms that `torch.as_tensor().cuda()` would spend even on unified memory (PyTorch always issues a memcpy regardless of `is_integrated`).

3. **One-time setup cost.** `cudaHostRegister` over 196 KB takes ~**1–2 ms** (vs. ~52 ms for 30 MB). Amortized over thousands of inference calls, this is negligible.

4. **Synchronization cost stays the same: ~1–5 µs** (POSIX semaphore, same as current design).

5. **No dependency conflict problem.** The shm approach requires both processes to map the same region — both can still live in separate venvs. The shared memory name (`/vla_obs`) is the only coupling. This is a drop-in replacement for ZMQ with no environment changes.

### What the async + chunking design adds (planned, not yet implemented)

The current shm design is still **synchronous** — the robot controller blocks waiting for each inference call. With the planned async + ring buffer design (Section 6 of this document):

6. **Robot never freezes.** The controller loop runs at 50 Hz consuming pre-computed action chunks. The inference server runs in parallel, posting new chunks as they finish. Even with 50–200 ms inference time, the robot executes smoothly.

7. **Inference server never idles.** As soon as a chunk is published to the ring buffer, the server starts the next inference immediately using the latest observation (double-buffered). There is no wait-for-client round trip.

8. **TRT engine writes actions directly into the ring buffer** via `context.set_tensor_address("actions", act_gpu_ptr)`. No intermediate tensor allocation, no decode step, no copy.

The net effect: a Jetson Orin running TRT KV+FBC inference (~20–30 ms per chunk) can sustain a 50 Hz control loop executing 8-step action chunks, with zero serialization and zero robot idle time.

### Can we do better than what's described here?

A few opportunities not yet explored:

**A. UNIX domain socket instead of TCP loopback.** Even staying with ZMQ, switching from `tcp://127.0.0.1:5555` to `ipc:///tmp/vla.sock` avoids the kernel TCP stack and reduces per-call overhead from ~1–2 ms to ~0.3–0.8 ms. This is a one-line change (`socket.bind("ipc:///tmp/vla.sock")`) and works with the existing `MsgSerializer`. Not as fast as shm but a zero-risk intermediate step.

**B. Zero-copy ZMQ message (`zmq.Frame` with `copy=False`).** ZMQ can avoid the internal buffer copy on send if the data is passed as a pre-allocated `zmq.Frame`. Combined with writing the numpy array directly into the frame buffer, this eliminates one memory copy. Still requires serialization, but removes one kernel copy per call. Useful as a stop-gap on non-Jetson hardware where shm doesn't give zero-copy.

**C. `cudaMallocHost` + `memfd_create` instead of `mmap` + `cudaHostRegister`.** The current design uses `cudaHostRegister` after the fact on an `mmap`'d region. Pre-allocating with `cudaMallocHost` and sharing the file descriptor via `memfd_create` + `SCM_RIGHTS` (Unix socket fd passing) gives cleaner pinning and avoids potential failures on large buffers or fragmented pages. This matters more as payload size grows (e.g., multi-camera setups).

**D. Pre-pinned numpy view on the client.** Currently the client writes into the shm region via a plain numpy `frombuffer` view — no special pinning. On Jetson this is fine since all memory is physically unified. On a future discrete-GPU server, pre-pinning the client-side buffer with `cudaHostRegister` (from the client process, not just the server) would allow DMA-direct transfer from client CPU write to GPU VRAM without a second kernel copy.

**E. Embedding the language once, not every call.** The language instruction is constant for an entire episode (typically 200–300 steps). In the current ZMQ design it is re-serialized and re-tokenized on every call. Moving language tokenization + embedding to an `episode_reset` path (triggered once) and caching the resulting `vl_embs` shape `[1, 122, 2048]` BF16 = ~50 KB in shm would save the backbone's attention over the text tokens — but this requires changes to `Gr00tPolicy._get_action()` to accept cached embeddings.
