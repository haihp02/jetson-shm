# GR00T VLA Transport Architecture: Current State & Future Work

## 1. Current Architecture (ZMQ REQ-REP)

### Overview

```
Client process (sim / real robot)         Server process (GPU inference)
┌──────────────────────────────┐          ┌────────────────────────────────┐
│ rollout_policy.py            │          │ run_gr00t_server.py            │
│   PolicyClient.get_action()  │ ── ZMQ ──►  PolicyServer.run()           │
│   (serializes obs dict)      │  REQ-REP  │  Gr00tSimPolicyWrapper         │
│                              │◄──ZMQ ───  │    → Gr00tPolicy._get_action() │
│   receives action chunk      │          │    → model.get_action()        │
└──────────────────────────────┘          └────────────────────────────────┘
                                          port 5555, tcp://0.0.0.0:5555
```

Two separate Python venvs (main GR00T env vs SimplerEnv) — dependency conflicts
prevent a single process, so ZMQ is the integration point between them.

### What is ZMQ (Quick Background)

ZeroMQ (ZMQ) is a messaging library that wraps BSD sockets with a higher-level pattern API. It is not a broker — there is no server process; it is just a thin layer on top of TCP/IPC sockets that handles framing, buffering, and reconnection.

The pattern used here is **REQ-REP** (request-reply):
- The client (REQ socket) calls `socket.send(data)` then blocks on `socket.recv()`.
- The server (REP socket) loops: `socket.recv()` → process → `socket.send(result)`.
- Strictly synchronous: one outstanding request at a time. The client cannot send a second request until it has received the reply.

On the same machine, ZMQ over TCP loopback (`tcp://127.0.0.1:5555`) goes through the kernel network stack: user space → syscall → kernel TCP stack → loopback → kernel TCP stack → syscall → user space. This is faster than a real network but still involves two context switches and a kernel copy per direction. The dominant cost here is not the socket overhead itself (~10–50 µs) but the **serialization**: every numpy array must be packed into bytes before `send()` and unpacked back after `recv()`.

### ZMQ Protocol

**Transport:** ZMQ REQ-REP, synchronous, single-threaded server (1 ThreadPoolExecutor worker), 90s timeout.

**Serialization:** `MsgSerializer` in `gr00t/policy/server_client.py`:
- `msgpack.packb` / `msgpack.unpackb` for the outer dict envelope
- numpy arrays encoded via `np.save()` into an in-memory `.npy` buffer and stored
  as `{"__ndarray_class__": True, "as_npy": <bytes>}` — each array gets its own
  numpy header (~128 bytes) plus raw data.

**Request envelope:**
```python
{
    "endpoint": "get_action",            # or ping / reset / get_modality_config / kill
    "data": {
        "observation": <obs_dict>,       # see below
        "options": None,
    }
}
```

**Registered endpoints:** `ping`, `kill`, `get_action`, `reset`, `get_modality_config`.

### Observation and Action Format (OXE_WIDOWX / SimplerEnv)

The client sends **flat-key** observations in `Gr00tSimPolicyWrapper` format:

| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `video.image_0` | `(B, 1, 256, 256, 3)` | uint8 | WidowX image, 1 temporal frame |
| `state.x` | `(B, 1, 1)` | float32 | EEF position |
| `state.y` | `(B, 1, 1)` | float32 | |
| `state.z` | `(B, 1, 1)` | float32 | |
| `state.roll` | `(B, 1, 1)` | float32 | Euler angle |
| `state.pitch` | `(B, 1, 1)` | float32 | |
| `state.yaw` | `(B, 1, 1)` | float32 | |
| `state.pad` | `(B, 1, 1)` | float32 | Always 0 |
| `state.gripper` | `(B, 1, 1)` | float32 | |
| `annotation.human.action.task_description` | `(B,)` | tuple[str] | Language instruction |

B = number of parallel sim environments (typically 5 in eval).

The server returns **flat-key** actions:

| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `action.x` | `(B, 8, 1)` | float32 | Action horizon = 8 for WidowX |
| `action.y` | `(B, 8, 1)` | float32 | |
| `action.z` | `(B, 8, 1)` | float32 | |
| `action.roll` | `(B, 8, 1)` | float32 | |
| `action.pitch` | `(B, 8, 1)` | float32 | |
| `action.yaw` | `(B, 8, 1)` | float32 | |
| `action.gripper` | `(B, 8, 1)` | float32 | |

Only the first `n_action_steps=4` of the 8-step chunk are actually executed per call.

### Observation → Server Pipeline

`Gr00tSimPolicyWrapper._get_action()` re-nests the flat keys before forwarding to `Gr00tPolicy`:

```
flat obs["video.image_0"]   →  nested obs["video"]["image_0"]     shape (B, 1, 256, 256, 3)
flat obs["state.x"] etc.    →  nested obs["state"]["x"] etc.      shape (B, 1, 1)
flat obs["annotation..."]   →  nested obs["language"]["annotation..."]  list[list[str]]
```

`Gr00tPolicy._get_action()`:
1. Unbatch: split `(B, T, ...)` into B individual observations
2. Convert each to `VLAStepData` (images + states + text + embodiment tag)
3. `processor(messages)` → tokenize, embed images
4. `collate_fn(processed_inputs)` → batch tensor
5. `model.get_action(**collated_inputs)` → `action_pred` shape `(B, 8, action_dim)`
6. `processor.decode_action()` → unnormalize back to physical units
7. Return `{key: array}` for each action key

### Data Volume per ZMQ Call (B=5, WidowX)

| Component | Raw bytes | Notes |
|-----------|-----------|-------|
| `video.image_0` | 5×1×256×256×3 = **983,040 B** (~960 KB) | dominant cost |
| All state arrays | 5×1×1×4 × 8 keys = **160 B** | negligible |
| Action response | 5×8×1×4 × 7 keys = **1,120 B** | negligible |
| msgpack + numpy headers | ~1 KB | per-array numpy headers |

Per-call overhead: serialize **~960 KB** of image data → loopback socket → deserialize → inference → serialize **~1 KB** action → loopback → deserialize.

Measured via `test_zmq_baseline.py` (200 iters, ZMQ tcp loopback, no model, dummy actions):

| | B=1 (real robot) | B=5 (sim eval) |
|---|---|---|
| Request payload | 193.6 KB | 962 KB |
| serialize | 0.30 ms | 1.29 ms |
| socket+copy | 1.41 ms | 2.27 ms |
| deserialize | 0.58 ms | 0.72 ms |
| **total round-trip** | **2.3 ms** | **4.3 ms** |

These are the pure transport numbers with no model inference. Add ~100–200 ms PyTorch BF16 inference on top (or ~20–50 ms with TRT KV+FBC). On Jetson the serialization cost will be similar or slightly higher due to slower CPU.

---

## 2. Model Architecture Summary

**Backbone:** VLM (large transformer, ~2B params, BF16)
- Input: tokenized images + text + state features
- Output: `vl_embs` shape `[1, 122, 2048]` — vision-language embeddings

**Action head:** DiT diffusion head
- Input: `vl_embs`, `actions_noise [1, 50, 128]`, `state_features [1, 1, 1536]`, attention masks
- Output: `final_actions` — 4 denoising steps (or 1 for distilled student)
- Produces action chunk of horizon 8 for WidowX (16 for other embodiments)

**TRT options (Orin deployment):**
- `kv` mode: KV-unfolded DiT, all 4 steps baked into one engine
- `kv_fbc` mode: KV + FBC baked in — fastest
- `full_pipeline`: backbone + DiT together
- Engine: ~2.1 GB file, ~778 MB runtime VRAM, 2141 TRT layers, 57% fusion ratio

---

## 3. Key Source Files

| File | Role |
|------|------|
| `gr00t/policy/server_client.py` | `PolicyServer` (ZMQ REP) + `PolicyClient` (ZMQ REQ) + `MsgSerializer` |
| `gr00t/policy/gr00t_policy.py` | `Gr00tPolicy` (core inference) + `Gr00tSimPolicyWrapper` (flat-key adapter) |
| `gr00t/policy/policy.py` | `BasePolicy` + `PolicyWrapper` abstract classes |
| `gr00t/eval/run_gr00t_server.py` | CLI entry point: instantiates policy + `PolicyServer`, runs `server.run()` |
| `gr00t/eval/run_gr00t_server_trt.py` | Same + TRT engine path routing |
| `gr00t/eval/rollout_policy.py` | Client side: `run_rollout_gymnasium_policy`, creates `PolicyClient` |
| `gr00t/eval/sim/SimplerEnv/simpler_env.py` | `WidowXBridgeEnv` gym wrapper — image 256×256, 8 state dims |
| `gr00t/eval/sim/wrapper/multistep_wrapper.py` | `MultiStepWrapper` — assembles `(B, T, ...)` obs from deque |
| `gr00t/configs/data/embodiment_configs.py` | Per-embodiment `ModalityConfig`: delta_indices, modality keys |
| `gr00t/eval/real_robot/SO100/eval_so100.py` | Real-robot eval — `So100Adapter` shows `PolicyClient` usage pattern |

---

## 4. Future Work

### Task A — ZMQ Baseline Test Script

Build `test_zmq_baseline.py` that mimics the current architecture **without** the full GR00T model:

- Spin up a `PolicyServer`-like ZMQ REP server that receives the real observation dict structure and returns dummy action chunks of the correct shape
- Spin up a `PolicyClient`-like REQ client that constructs synthetic WidowX observations of the correct shape (`video.image_0: (1, 1, 256, 256, 3)`, 8 state keys, language string) and calls `get_action` in a tight loop
- Measure: per-call latency (p50/p95/p99), serialization time, deserialization time, total round-trip
- Profile with nsys to see where CPU time goes (msgpack encode/decode vs socket overhead)
- This gives us the **baseline transport cost** independent of model inference time

Key things to match from the real system:
- Use `msgpack` + `np.save()` exactly as `MsgSerializer` does (not raw bytes)
- Match real payload sizes: `(1, 1, 256, 256, 3)` uint8 image = ~196 KB, 8 × `(1, 1, 1)` float32 states
- Simulate batch size B=1 (real robot) and B=5 (sim eval)

### Task B — Shared Memory Test Script Refinement ✓ DONE (`test_shm_vla.py`)

**Results on Orin** — all three tests pass:
- Test A: `img_gpu.sum(dtype=cp.int64)` = 25,067,520 matches CPU-computed expected ✓
- Test B: float32 action write/read across processes ✓
- Test C: 100-iteration obs→GPU-read→act loop completes ✓

Key findings from the run:
- `cudaHostRegister` works correctly on a mixed-dtype (uint8 + float32) shm region — no special handling needed for different dtypes
- Per-section CuPy views via `UnownedMemory(gpu_ptr.value + byte_offset, size)` work correctly for both uint8 (image) and float32 (action) sections
- `img_gpu.sum(dtype=cp.int64)` accumulates in int64 — no overflow, correct result
- CPU ptr ≠ GPU ptr (CUDA driver uses separate VA ranges) but `is_integrated=True` confirms same physical DRAM
- Language stored as null-padded bytes at a fixed offset; no GPU access needed
- Shm size: 197,440 B (192.8 KB) vs test_shm.py's 9.8 MB (SHAPE=(8,480,640) float32) — 50× smaller

Adapt `test_shm.py` to match the actual GR00T VLA payload:

**Observation buffer layout:**

The current `test_shm.py` uses a single flat float32 array of shape `(8, 480, 640)`. The real VLA payload needs:

| Field | Shape | Dtype | Bytes |
|-------|-------|-------|-------|
| `image_0` | `(1, 256, 256, 3)` | uint8 | 196,608 |
| `state` (8 values) | `(8,)` | float32 | 32 |
| `action` (7 × 8) | `(56,)` | float32 | 224 |
| Header / semaphore flags | `(4,)` | int32 | 16 |
| **Total** | | | **≈ 197 KB** |

Changes needed in `test_shm.py`:
- Replace single float32 tensor with a structured layout: header + uint8 image + float32 state + float32 action
- The GPU kernel (currently `cp_array.sum()`) should be replaced by something that reads all bytes of the image: e.g. `cp_array.view(cp.uint8).sum()` or a small reduction over the uint8 image buffer
- Test correctness with uint8 image data (client writes random pixels, server GPU reads via CuPy, verifies sum)
- Keep the semaphore synchronization protocol as-is (it already matches the VLA obs/act ping-pong)
- Add a `task_description` field if needed — language is a string, not in the GPU-accessible buffer; can be a separate small fixed-size char array or passed out-of-band

Note: CuPy does not natively handle uint8 reductions with the same path as float32 — verify correctness that `cudaHostRegister` works correctly on the mixed-dtype layout.

### Task C — Benchmarking and Comparison ✓ DONE

Measured on Jetson Orin. ZMQ via `test_zmq_baseline.py`, shm via `test_shm_vla.py` + `nsys profile`.

#### Results (p50, B=1, no model inference)

| Component | ZMQ (B=1) | shm |
|---|---|---|
| Data preparation | 297 µs (msgpack + np.save) | 265 µs* (rng + write) |
| Transport core | 1,414 µs (socket + kernel copy) | 205 µs (semaphore pair + GPU kernel) |
| Decode | 578 µs (np.load + unpack) | — |
| **Total (p50)** | **2,242 µs** | **205 µs** |
| **Total (p99)** | **2,856 µs** | **230 µs** |
| **Speedup** | — | **10.9×** |
| `cudaHostRegister` one-time | N/A | 56 ms (nsys measured) |

*The shm obs_write (265 µs) is dominated by `rng.integers()` generating random test data, not the memory write itself. In production the camera frame is already in RAM; copying 192 KB costs ~50 µs. ZMQ's 297 µs serialize is a real production cost with no shm equivalent.

#### shm round-trip breakdown (from nsys, per iteration)

| Sub-component | Time |
|---|---|
| `cuLaunchKernel` × 2 (CPU-side) | 30 µs |
| GPU kernels (CUB sum pass1 + pass2) | 14 µs |
| `cudaMemcpyAsync` D2H (fetch `int()` result) | 25 µs |
| `cudaStreamSynchronize` | 3 µs |
| Semaphore ops + OS scheduling (2 context switches) | ~146 µs |
| **Total** | **~205 µs** |

The dominant cost is **OS scheduling** (~146 µs for two cross-process context switches on Orin), not GPU or data movement. The GPU only does 14 µs of real work on the 192 KB image. The 25 µs `cudaMemcpyAsync` to fetch the scalar sum back to Python is test overhead — in real TRT inference the server never reads a value back; actions are written directly into the shm action buffer. Estimated real TRT round-trip: **~177 µs** (~12.7× vs ZMQ).

#### What the speedup means in practice

| Scenario | Inference | ZMQ overhead | shm overhead | shm saves |
|---|---|---|---|---|
| PyTorch BF16, 10 Hz | ~150 ms | 2.2 ms (1.5%) | 0.2 ms | ~2 ms |
| TRT KV+FBC, 30 Hz | ~30 ms | 2.2 ms (7%) | 0.2 ms | ~2 ms |
| TRT KV+FBC, tight 50 Hz | ~20 ms | 2.2 ms (11%) | 0.2 ms | **~2 ms** |
| Async chunking (planned) | hidden | **2.2 ms blocks robot** | **0 (non-blocking)** | **entire call** |

The transport saving is ~2 ms absolute, which matters proportionally once TRT drives inference below 30 ms. The structural difference is in the async design: ZMQ's synchronous REQ-REP freezes the control loop on every call; shm's ring buffer makes the transport non-blocking entirely.

---

## 5. Open Questions

1. **TRT zero-copy output:** Can `context.set_tensor_address()` write action predictions directly into the shm action buffer? This would eliminate the last remaining copy in the hot loop.

2. **String / language via shm:** Language instructions are strings and change infrequently (once per episode). Options: (a) fixed-size null-padded char array in the header, (b) small separate ZMQ channel only for episode resets, (c) language embedding pre-computed and stored in shm.

3. **Batch size > 1 in shm:** For sim eval with B=5 parallel envs, each env would need its own shm region, or a single region with strided per-env slots. The current design is single-producer-single-consumer; extending to B=5 requires care.

4. **CuPy vs TRT on Orin:** For the actual inference server, CuPy's `cp_array.sum()` will be replaced by a TRT engine call. The shm integration point is `context.set_tensor_address("obs", obs_gpu_ptr)` — verify TRT on SM 8.7 (Orin) accepts the `cudaHostGetDevicePointer` pointer type.
