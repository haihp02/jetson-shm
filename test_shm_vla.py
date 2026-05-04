"""
test_shm_vla.py
---------------
Shared-memory transport test using the actual GR00T OXE_WIDOWX payload layout.

Replaces test_shm.py's synthetic (8,480,640) float32 = 30 MB tensor with a
structured shm region matching the real VLA obs/action sizes:

  [ header 64B ] [ image_0 196608B uint8 ] [ state 32B float32 ] [ action 224B float32 ] [ lang 512B ]
  Total: 197,440 B (~193 KB)

The key difference from test_shm.py:
  - Mixed dtypes (uint8 image + float32 state/action) in one shm region
  - Per-section CuPy views via gpu_ptr + byte offsets
  - GPU op: img_gpu.sum(dtype=int64) reads the 192 KB image (vs 30 MB before)
  - Language stored as null-padded bytes; CPU-only, no GPU access needed

Tests:
  A — client writes sequential image bytes via CPU  →  server reads sum via GPU
  B — server writes action=42.0 via CPU            →  client verifies
  C — 100-iteration loop: client obs-write → server GPU-read → CPU act-write → client act-read

Run:
  python test_shm_vla.py
  nsys profile --trace=cuda python test_shm_vla.py
"""

import ctypes
import mmap
import multiprocessing as mp
import sys
import time

import cupy as cp
import numpy as np
import posix_ipc
import torch

# ── buffer layout ──────────────────────────────────────────────────────────────
HDR_BYTES    = 64                        # 16 × int32, reserved / future control fields
IMG_SHAPE    = (1, 256, 256, 3)          # OXE_WIDOWX: 1 temporal frame, 256×256 RGB
IMG_BYTES    = int(np.prod(IMG_SHAPE))   # 196,608 B
STATE_DIM    = 8                         # x y z roll pitch yaw pad gripper
STATE_BYTES  = STATE_DIM * 4            # 32 B
ACTION_DIM   = 7 * 8                    # 7 keys × action_horizon=8 steps = 56 floats
ACTION_BYTES = ACTION_DIM * 4          # 224 B
LANG_BYTES   = 512                       # null-padded UTF-8

IMG_OFF      = HDR_BYTES                # 64
STATE_OFF    = IMG_OFF    + IMG_BYTES   # 196,672  (4-byte aligned ✓)
ACTION_OFF   = STATE_OFF  + STATE_BYTES # 196,704  (4-byte aligned ✓)
LANG_OFF     = ACTION_OFF + ACTION_BYTES# 196,928
NBYTES       = LANG_OFF   + LANG_BYTES  # 197,440

# ── IPC names ──────────────────────────────────────────────────────────────────
SHM_NAME    = "/vla_shm_vla"
SEM_A_NAME  = "/vla_sem_va"
SEM_B_NAME  = "/vla_sem_vb"
SEM_C0_NAME = "/vla_sem_vc0"   # client → server (obs ready)
SEM_C1_NAME = "/vla_sem_vc1"   # server → client (act ready)
N_WARMUP    = 5
N_ITERS     = 100
LANG_INSTR  = "Pick up the eggplant and put it in the basket."
# ───────────────────────────────────────────────────────────────────────────────


# ── CPU views ──────────────────────────────────────────────────────────────────
def make_cpu_views(mem: mmap.mmap):
    """Typed numpy views for each section, all backed by the same mmap."""
    raw   = np.ndarray(NBYTES, dtype=np.uint8, buffer=mem)
    img   = raw[IMG_OFF   : IMG_OFF   + IMG_BYTES  ].reshape(IMG_SHAPE)
    state = raw[STATE_OFF : STATE_OFF + STATE_BYTES ].view(np.float32)
    act   = raw[ACTION_OFF: ACTION_OFF + ACTION_BYTES].view(np.float32)
    lang  = raw[LANG_OFF  : LANG_OFF  + LANG_BYTES  ]
    return img, state, act, lang


# ── CUDA registration ──────────────────────────────────────────────────────────
def register_cuda(mem: mmap.mmap):
    """Pin the shm region with cudaHostRegister, return (cudart, cpu_ptr, gpu_ptr)."""
    torch.cuda.init()
    cudart  = ctypes.CDLL("libcudart.so")
    cpu_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mem)))
    ret = cudart.cudaHostRegister(cpu_ptr, NBYTES, 0x02)
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed: {ret}")
    gpu_ptr = ctypes.c_void_p()
    ret = cudart.cudaHostGetDevicePointer(ctypes.byref(gpu_ptr), cpu_ptr, 0)
    if ret != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {ret}")
    return cudart, cpu_ptr, gpu_ptr


# ── GPU views ──────────────────────────────────────────────────────────────────
def make_gpu_views(gpu_ptr: ctypes.c_void_p):
    """CuPy array views at the image and action offsets of the GPU mapping."""
    def _view(byte_offset, shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        umem   = cp.cuda.UnownedMemory(gpu_ptr.value + byte_offset, nbytes, owner=None)
        return cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(umem, 0))

    img_gpu = _view(IMG_OFF,    IMG_SHAPE,     cp.uint8)
    act_gpu = _view(ACTION_OFF, (ACTION_DIM,), cp.float32)
    return img_gpu, act_gpu


# ── server process ─────────────────────────────────────────────────────────────
def server(addr_queue: mp.Queue):
    print("[server] starting")
    shm    = posix_ipc.SharedMemory(SHM_NAME, flags=posix_ipc.O_CREAT, size=NBYTES)
    mem    = mmap.mmap(shm.fd, NBYTES)
    sem_a  = posix_ipc.Semaphore(SEM_A_NAME,  flags=posix_ipc.O_CREAT, initial_value=0)
    sem_b  = posix_ipc.Semaphore(SEM_B_NAME,  flags=posix_ipc.O_CREAT, initial_value=0)
    sem_c0 = posix_ipc.Semaphore(SEM_C0_NAME, flags=posix_ipc.O_CREAT, initial_value=0)
    sem_c1 = posix_ipc.Semaphore(SEM_C1_NAME, flags=posix_ipc.O_CREAT, initial_value=0)
    cudart = cpu_ptr = None

    try:
        cudart, cpu_ptr, gpu_ptr = register_cuda(mem)
        _, _, act_cpu, _         = make_cpu_views(mem)
        img_gpu, _               = make_gpu_views(gpu_ptr)

        print("[server] shm registered with CUDA ✓")
        addr_queue.put({"server_cpu": cpu_ptr.value, "server_gpu": gpu_ptr.value})

        # ── Test A: client writes image via CPU → server reads via GPU ─────────
        print("[server] TEST A — waiting for client image write ...")
        sem_a.acquire()
        gpu_sum = int(img_gpu.sum(dtype=cp.int64))
        # expected: sum of [0,1,...,255] repeated 768 times = 768 × 32640
        exp_sum = int(np.arange(IMG_BYTES, dtype=np.uint8).astype(np.int64).sum())
        if gpu_sum == exp_sum:
            print(f"[server] TEST A PASSED ✓  GPU sum={gpu_sum} matches CPU-computed expected\n")
        else:
            print(f"[server] TEST A FAILED ✗  GPU sum={gpu_sum}  expected={exp_sum}\n")

        # ── Test B: server writes action via CPU → client reads ─────────────────
        print("[server] TEST B — writing action=42.0 via CPU ...")
        act_cpu[:] = 42.0
        sem_b.release()

        # ── Test C: obs/act loop (profiled with nsys) ───────────────────────────
        print(f"[server] TEST C — {N_WARMUP + N_ITERS} iterations ({N_WARMUP} warm-up) ...")
        for i in range(N_WARMUP + N_ITERS):
            sem_c0.acquire()
            # int() forces CUDA sync — mirrors what TRT inference does before returning
            _ = int(img_gpu.sum(dtype=cp.int64))
            act_cpu[:] = float(i % 256)
            sem_c1.release()
        print("[server] TEST C DONE ✓\n")

    finally:
        if cudart is not None and cpu_ptr is not None:
            cudart.cudaHostUnregister(cpu_ptr)
        mem.close()
        shm.unlink()
        for s in [sem_a, sem_b, sem_c0, sem_c1]:
            s.unlink()
        print("[server] cleaned up")


# ── client process ─────────────────────────────────────────────────────────────
def client(addr_queue: mp.Queue):
    time.sleep(0.5)
    print("[client] starting")
    shm    = posix_ipc.SharedMemory(SHM_NAME)
    mem    = mmap.mmap(shm.fd, NBYTES)
    sem_a  = posix_ipc.Semaphore(SEM_A_NAME)
    sem_b  = posix_ipc.Semaphore(SEM_B_NAME)
    sem_c0 = posix_ipc.Semaphore(SEM_C0_NAME)
    sem_c1 = posix_ipc.Semaphore(SEM_C1_NAME)
    img_cpu, state_cpu, act_cpu, lang_cpu = make_cpu_views(mem)
    cpu_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mem)))
    addr_queue.put({"client_cpu": cpu_ptr.value})

    # Write language instruction once (constant for the episode)
    enc = LANG_INSTR.encode("utf-8")
    lang_cpu[:] = 0
    lang_cpu[:len(enc)] = np.frombuffer(enc, dtype=np.uint8)

    # ── Test A ──────────────────────────────────────────────────────────────────
    print("[client] TEST A — writing sequential image bytes via CPU ...")
    img_cpu[:] = np.arange(IMG_BYTES, dtype=np.uint8).reshape(IMG_SHAPE)
    sem_a.release()

    # ── Test B ──────────────────────────────────────────────────────────────────
    sem_b.acquire()
    if np.all(act_cpu == 42.0):
        print("[client] TEST B PASSED ✓  sees server action write\n")
    else:
        print("[client] TEST B FAILED ✗\n")

    # ── Test C ──────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    print(f"[client] TEST C — {N_WARMUP} warm-up + {N_ITERS} timed iterations ...")
    records = []
    for i in range(N_WARMUP + N_ITERS):
        t0 = time.perf_counter()
        img_cpu[:]   = rng.integers(0, 256, IMG_SHAPE, dtype=np.uint8)
        state_cpu[:] = rng.standard_normal(STATE_DIM).astype(np.float32)
        t1 = time.perf_counter()
        sem_c0.release()
        sem_c1.acquire()
        t2 = time.perf_counter()
        if i >= N_WARMUP:
            records.append({
                "obs_write_us":  (t1 - t0) * 1e6,
                "round_trip_us": (t2 - t1) * 1e6,
                "total_us":      (t2 - t0) * 1e6,
            })
    print("[client] TEST C DONE ✓\n")
    addr_queue.put({"timing": records})

    mem.close()
    print("[client] done")


# ── timing report ──────────────────────────────────────────────────────────────
def print_timing(records: list):
    a = {k: np.array([r[k] for r in records]) for k in records[0]}

    def row(label, v):
        print(f"  {label:<22} {np.mean(v):>8.1f} {np.percentile(v,50):>8.1f} "
              f"{np.percentile(v,95):>8.1f} {np.percentile(v,99):>8.1f} {np.max(v):>8.1f}")

    print(f"\n── Test C latency  ({len(records)} iters, {N_WARMUP} warm-up discarded) ──────────")
    print(f"  {'component':<22} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}  µs")
    print("  " + "-" * 68)
    row("obs write (CPU→shm)",  a["obs_write_us"])
    row("round-trip (sem+GPU)", a["round_trip_us"])
    row("total",                a["total_us"])

    zmq_b1_us = 2242   # test_zmq_baseline.py p50, B=1
    shm_p50   = float(np.percentile(a["total_us"], 50))
    print(f"\n── vs ZMQ (B=1, p50) ──────────────────────────────────────────")
    print(f"  ZMQ total    : {zmq_b1_us:>8} µs")
    print(f"  shm total    : {shm_p50:>8.1f} µs")
    print(f"  speedup      : {zmq_b1_us / shm_p50:>8.1f}×")


# ── layout / pointer report ────────────────────────────────────────────────────
def print_report(addrs: dict):
    server_cpu = addrs.get("server_cpu")
    server_gpu = addrs.get("server_gpu")
    client_cpu = addrs.get("client_cpu")
    props      = torch.cuda.get_device_properties(0)
    unified    = bool(props.is_integrated)

    old_bytes = 8 * 480 * 640 * 4  # test_shm.py payload

    print("\n── Buffer layout ──────────────────────────────────────────────")
    print(f"  Total           : {NBYTES:,} B  ({NBYTES/1024:.1f} KB)")
    print(f"  header (int32)  : {HDR_BYTES} B    @ +0")
    print(f"  image_0 (uint8) : {IMG_BYTES:,} B  @ +{IMG_OFF:<7}  shape={IMG_SHAPE}")
    print(f"  state  (f32)    : {STATE_BYTES} B    @ +{STATE_OFF:<7}  {STATE_DIM} values")
    print(f"  action (f32)    : {ACTION_BYTES} B   @ +{ACTION_OFF:<7}  7 keys × 8 steps = {ACTION_DIM} values")
    print(f"  language (utf8) : {LANG_BYTES} B   @ +{LANG_OFF}")

    print("\n── Pointers ───────────────────────────────────────────────────")
    print(f"  Device          : {props.name}  (is_integrated={unified})")
    print(f"  server CPU ptr  : 0x{server_cpu:016x}")
    print(f"  server GPU ptr  : 0x{server_gpu:016x}")
    print(f"  client CPU ptr  : 0x{client_cpu:016x}")

    if server_cpu == server_gpu:
        print("\n  ✓ CPU ptr == GPU ptr  (unified address space)")
    else:
        print("\n  ~ CPU ptr != GPU ptr  (CUDA driver maps to separate VA ranges — expected)")

    if unified:
        print("  ✓ ZERO-COPY  — is_integrated=True, CPU & GPU share physical DRAM")
    else:
        print("  ✗ NOT ZERO-COPY — discrete GPU, GPU reads stall on PCIe per cache miss")

    print(f"\n── vs test_shm.py ─────────────────────────────────────────────")
    print(f"  old payload : (8,480,640) float32 = {old_bytes/1024/1024:.0f} MB")
    print(f"  this payload: {IMG_SHAPE} uint8  = {IMG_BYTES/1024:.0f} KB")
    print(f"  {old_bytes // IMG_BYTES}× smaller  →  GPU kernel ~1 µs vs ~123 µs (Orin, test_shm.py)")
    print("───────────────────────────────────────────────────────────────\n")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Clean up any leftover IPC objects from a previous crashed run
    for name in [SHM_NAME]:
        try: posix_ipc.SharedMemory(name).unlink()
        except: pass
    for name in [SEM_A_NAME, SEM_B_NAME, SEM_C0_NAME, SEM_C1_NAME]:
        try: posix_ipc.Semaphore(name).unlink()
        except: pass

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"Device    : {torch.cuda.get_device_name(0)}")
    print(f"Shm size  : {NBYTES:,} B  ({NBYTES/1024:.1f} KB)")
    print(f"Iterations: {N_ITERS}\n")
    print("=" * 55)

    addr_queue = mp.Queue()
    p_server   = mp.Process(target=server, args=(addr_queue,))
    p_client   = mp.Process(target=client, args=(addr_queue,))
    p_server.start()
    p_client.start()
    p_server.join()
    p_client.join()

    addrs  = {}
    timing = []
    while not addr_queue.empty():
        item = addr_queue.get()
        if "timing" in item:
            timing = item["timing"]
        else:
            addrs.update(item)

    print("=" * 55)
    if timing:
        print_timing(timing)
    print_report(addrs)
    print("Done.")
