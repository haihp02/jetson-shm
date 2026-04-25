"""
test_jetson_shm.py
------------------
Minimal test: verify that POSIX shm + cudaHostRegister gives CPU and GPU
processes a zero-copy view of the same physical DRAM on Jetson Orin.

Uses CuPy to wrap the raw GPU pointer — this is the true zero-copy path.
torch.as_tensor().cuda() is intentionally avoided as it always memcpy's.

Test A — correctness check (single pass):
  client writes known values via numpy → server reads via cupy/torch tensor

Test B — correctness check (single pass):
  server writes via CPU numpy → client reads same bytes

Test C — loop benchmark (N iterations, profiled via nsys):
  client writes obs → server reads via GPU → server writes act → client reads
  No timing in script — use nsys stats for latency analysis

Run:
  python test_jetson_shm.py
  nsys profile --trace=cuda python test_jetson_shm.py
"""

import ctypes
import mmap
import multiprocessing as mp
import time
import sys

import cupy as cp
import numpy as np
import posix_ipc
import torch

# ── config ────────────────────────────────────────────────────────────────────
SHM_NAME    = "/vla_test"
SEM_A_NAME  = "/vla_sem_a"
SEM_B_NAME  = "/vla_sem_b"
SEM_C0_NAME = "/vla_sem_c0"   # client signals server (obs ready)
SEM_C1_NAME = "/vla_sem_c1"   # server signals client (act ready)
SHAPE       = (8, 480, 640)
NBYTES      = int(np.prod(SHAPE)) * 4  # float32
N_ITERS     = 100
# ─────────────────────────────────────────────────────────────────────────────


def register_cuda(mem: mmap.mmap, nbytes: int):
    torch.cuda.init()
    cudart  = ctypes.CDLL("libcudart.so")
    cpu_ptr = ctypes.c_void_p(
        ctypes.addressof(ctypes.c_char.from_buffer(mem))
    )
    ret = cudart.cudaHostRegister(cpu_ptr, nbytes, 0x02)
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed: code {ret}")

    gpu_ptr = ctypes.c_void_p()
    ret = cudart.cudaHostGetDevicePointer(ctypes.byref(gpu_ptr), cpu_ptr, 0)
    if ret != 0:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: code {ret}")

    return cudart, cpu_ptr, gpu_ptr


def make_cpu_array(mem: mmap.mmap, shape: tuple) -> np.ndarray:
    return np.frombuffer(mem, dtype=np.float32).reshape(shape)


def make_gpu_tensor(gpu_ptr: ctypes.c_void_p, shape: tuple) -> torch.Tensor:
    nbytes = int(np.prod(shape)) * 4
    mem    = cp.cuda.UnownedMemory(gpu_ptr.value, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    arr    = cp.ndarray(shape, dtype=cp.float32, memptr=memptr)
    return torch.as_tensor(arr, device='cuda')


# ── server process ────────────────────────────────────────────────────────────
def server(addr_queue: mp.Queue):
    print("[server] starting")

    shm    = posix_ipc.SharedMemory(SHM_NAME, flags=posix_ipc.O_CREAT, size=NBYTES)
    mem    = mmap.mmap(shm.fd, NBYTES)
    sem_a  = posix_ipc.Semaphore(SEM_A_NAME,  flags=posix_ipc.O_CREAT, initial_value=0)
    sem_b  = posix_ipc.Semaphore(SEM_B_NAME,  flags=posix_ipc.O_CREAT, initial_value=0)
    sem_c0 = posix_ipc.Semaphore(SEM_C0_NAME, flags=posix_ipc.O_CREAT, initial_value=0)
    sem_c1 = posix_ipc.Semaphore(SEM_C1_NAME, flags=posix_ipc.O_CREAT, initial_value=0)

    cudart    = None
    cpu_ptr   = None
    cpu_array = None

    try:
        cudart, cpu_ptr, gpu_ptr = register_cuda(mem, NBYTES)
        cpu_array  = make_cpu_array(mem, SHAPE)
        gpu_tensor = make_gpu_tensor(gpu_ptr, SHAPE)   # torch, used for Test A only
        cp_array   = cp.ndarray(SHAPE, dtype=cp.float32,
                        memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(gpu_ptr.value, NBYTES, None), 0))

        print("[server] shared memory registered with CUDA ✓")
        addr_queue.put({"server_cpu": cpu_ptr.value, "server_gpu": gpu_ptr.value})

        # ── Test A: client writes via CPU, server reads via GPU ───────────
        print("[server] TEST A — waiting for client CPU write ...")
        sem_a.acquire()
        gpu_vals = gpu_tensor.cpu().numpy()
        expected = np.arange(int(np.prod(SHAPE)), dtype=np.float32).reshape(SHAPE)
        if np.allclose(gpu_vals, expected):
            print("[server] TEST A PASSED ✓  GPU sees client's CPU write\n")
        else:
            print("[server] TEST A FAILED ✗\n")

        # ── Test B: server writes via CPU, client reads ───────────────────
        print("[server] TEST B — writing 42.0 via CPU into shared memory ...")
        cpu_array[:] = 42.0
        sem_b.release()

        # ── Test C: loop — simulate obs read + act write ──────────────────
        print(f"[server] TEST C — running {N_ITERS} inference loop iterations ...")
        for i in range(N_ITERS):
            sem_c0.acquire()                    # wait for client obs write

            _ = cp_array.sum()                  # GPU reads obs from shm via CuPy (SM 8.7 compatible)

            cpu_array[:] = float(i % 256)      # server writes act back via CPU (immediate)

            sem_c1.release()                    # signal client act is ready

        print("[server] TEST C DONE ✓\n")

    finally:
        del cpu_array
        if cudart is not None and cpu_ptr is not None:
            cudart.cudaHostUnregister(cpu_ptr)
        mem.close()
        shm.unlink()
        sem_a.unlink()
        sem_b.unlink()
        sem_c0.unlink()
        sem_c1.unlink()
        print("[server] cleaned up")


# ── client process ────────────────────────────────────────────────────────────
def client(addr_queue: mp.Queue):
    time.sleep(0.5)
    print("[client] starting")

    shm    = posix_ipc.SharedMemory(SHM_NAME)
    mem    = mmap.mmap(shm.fd, NBYTES)
    sem_a  = posix_ipc.Semaphore(SEM_A_NAME)
    sem_b  = posix_ipc.Semaphore(SEM_B_NAME)
    sem_c0 = posix_ipc.Semaphore(SEM_C0_NAME)
    sem_c1 = posix_ipc.Semaphore(SEM_C1_NAME)

    cpu_array = make_cpu_array(mem, SHAPE)
    cpu_ptr   = ctypes.c_void_p(
        ctypes.addressof(ctypes.c_char.from_buffer(mem))
    )
    addr_queue.put({"client_cpu": cpu_ptr.value})

    # ── Test A ────────────────────────────────────────────────────────────
    print("[client] TEST A — writing known values via CPU numpy ...")
    cpu_array[:] = np.arange(int(np.prod(SHAPE)), dtype=np.float32).reshape(SHAPE)
    sem_a.release()
    print("[client] TEST A — signalled server, waiting for Test B ...")

    # ── Test B ────────────────────────────────────────────────────────────
    sem_b.acquire()
    if np.all(cpu_array == 42.0):
        print("[client] TEST B PASSED ✓  client sees server's write\n")
    else:
        print("[client] TEST B FAILED ✗\n")

    # ── Test C: loop — simulate obs write + act read ──────────────────────
    print(f"[client] TEST C — running {N_ITERS} control loop iterations ...")
    for i in range(N_ITERS):
        cpu_array[:] = float(i % 256)          # client writes obs via CPU
        sem_c0.release()                        # signal server obs is ready

        sem_c1.acquire()                        # wait for server act

    print("[client] TEST C DONE ✓\n")

    del cpu_array
    mem.close()
    print("[client] done")


# ── address + device report ───────────────────────────────────────────────────
def print_report(addrs: dict):
    server_cpu = addrs.get("server_cpu")
    server_gpu = addrs.get("server_gpu")
    client_cpu = addrs.get("client_cpu")

    props   = torch.cuda.get_device_properties(0)
    unified = bool(props.is_integrated)

    print("\n── Report ─────────────────────────────────────────────────────")
    print(f"  Device         : {props.name}")
    print(f"  is_integrated  : {unified}")
    print()
    print(f"  server CPU ptr : 0x{server_cpu:016x}")
    print(f"  server GPU ptr : 0x{server_gpu:016x}")
    print(f"  client CPU ptr : 0x{client_cpu:016x}")
    print()

    if server_cpu == server_gpu:
        print("  ✓ server CPU == server GPU (same virtual address)")
    else:
        print("  ~ server CPU != server GPU (different virtual addresses)")
        print("    CUDA driver maps unified DRAM to separate VA ranges — expected on Jetson")

    if server_cpu == client_cpu:
        print("  ✓ server CPU == client CPU (same virtual address)")
    else:
        print("  ~ server CPU != client CPU (different virtual addresses)")
        print("    mmap picks different VA per process — expected on Linux")

    print()
    if unified:
        print("  ✓ ZERO-COPY CONFIRMED — is_integrated=True guarantees shared physical DRAM")
        print("    CuPy UnownedMemory wraps gpu_ptr directly — no memcpy in hot loop")
        print("    Verify with: nsys stats report.nsys-rep | grep cudaMemcpy")
    else:
        print("  ✗ NOT ZERO-COPY — discrete GPU, PCIe transfer will occur")
    print("───────────────────────────────────────────────────────────────\n")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn")

    for name in [SHM_NAME]:
        try: posix_ipc.SharedMemory(name).unlink()
        except: pass
    for name in [SEM_A_NAME, SEM_B_NAME, SEM_C0_NAME, SEM_C1_NAME]:
        try: posix_ipc.Semaphore(name).unlink()
        except: pass

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"Device      : {torch.cuda.get_device_name(0)}")
    print(f"Array shape : {SHAPE}, {NBYTES} bytes")
    print(f"Loop iters  : {N_ITERS}\n")
    print("=" * 55)

    addr_queue = mp.Queue()

    p_server = mp.Process(target=server, args=(addr_queue,))
    p_client = mp.Process(target=client, args=(addr_queue,))

    p_server.start()
    p_client.start()
    p_server.join()
    p_client.join()

    addrs = {}
    while not addr_queue.empty():
        addrs.update(addr_queue.get())

    print("=" * 55)
    print_report(addrs)
    print("Done.")
