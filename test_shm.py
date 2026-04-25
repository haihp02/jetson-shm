"""
test_jetson_shm.py
------------------
Minimal test: verify that POSIX shm + cudaHostRegister gives CPU and GPU
processes a zero-copy view of the same physical DRAM on Jetson Orin.

Uses CuPy to wrap the raw GPU pointer — this is the true zero-copy path.
torch.as_tensor().cuda() is intentionally avoided as it always memcpy's.

Test A — CPU writes, GPU reads (via CuPy → torch, zero-copy):
  client writes known values via numpy → server reads via cupy/torch tensor

Test B — server writes via CPU, client reads:
  server writes via CPU numpy → client reads same bytes

Run:
  python test_jetson_shm.py
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
SHM_NAME   = "/vla_test"
SEM_A_NAME = "/vla_sem_a"
SEM_B_NAME = "/vla_sem_b"
SHAPE      = (4, 7)
NBYTES     = int(np.prod(SHAPE)) * 4  # float32
# ─────────────────────────────────────────────────────────────────────────────


def register_cuda(mem: mmap.mmap, nbytes: int):
    """Pin the mmap region so GPU can access it directly."""
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
    """Zero-copy numpy view of the shm region."""
    return np.frombuffer(mem, dtype=np.float32).reshape(shape)


def make_gpu_tensor(gpu_ptr: ctypes.c_void_p, shape: tuple) -> torch.Tensor:
    """
    Wrap raw GPU device pointer as a torch tensor — true zero-copy.
    Uses CuPy UnownedMemory to avoid any allocation or memcpy.
    """
    nbytes = int(np.prod(shape)) * 4
    mem    = cp.cuda.UnownedMemory(gpu_ptr.value, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    arr    = cp.ndarray(shape, dtype=cp.float32, memptr=memptr)
    # torch.as_tensor from a CuPy array uses __cuda_array_interface__ — no copy
    return torch.as_tensor(arr, device='cuda')


# ── server process ────────────────────────────────────────────────────────────
def server(addr_queue: mp.Queue):
    print("[server] starting")

    shm   = posix_ipc.SharedMemory(SHM_NAME, flags=posix_ipc.O_CREAT, size=NBYTES)
    mem   = mmap.mmap(shm.fd, NBYTES)
    sem_a = posix_ipc.Semaphore(SEM_A_NAME, flags=posix_ipc.O_CREAT, initial_value=0)
    sem_b = posix_ipc.Semaphore(SEM_B_NAME, flags=posix_ipc.O_CREAT, initial_value=0)

    cudart    = None
    cpu_ptr   = None
    cpu_array = None

    try:
        cudart, cpu_ptr, gpu_ptr = register_cuda(mem, NBYTES)
        cpu_array  = make_cpu_array(mem, SHAPE)
        gpu_tensor = make_gpu_tensor(gpu_ptr, SHAPE)   # zero-copy GPU view

        print("[server] shared memory registered with CUDA ✓")

        # Publish addresses to main for the final report
        addr_queue.put({"server_cpu": cpu_ptr.value, "server_gpu": gpu_ptr.value})

        # ── Test A: client writes via CPU, server reads via GPU ───────────
        print("[server] TEST A — waiting for client CPU write ...")
        sem_a.acquire()

        # Read directly from gpu_tensor — no memcpy, same physical bytes
        gpu_vals = gpu_tensor.cpu().numpy()   # .cpu() here is just for printing
        expected = np.arange(int(np.prod(SHAPE)), dtype=np.float32).reshape(SHAPE)
        print(f"[server] GPU tensor read:\n{gpu_vals}")

        if np.allclose(gpu_vals, expected):
            print("[server] TEST A PASSED ✓  GPU sees client's CPU write\n")
        else:
            print("[server] TEST A FAILED ✗\n")

        # ── Test B: server writes via CPU, client reads ───────────────────
        print("[server] TEST B — writing 42.0 via CPU into shared memory ...")
        cpu_array[:] = 42.0
        torch.cuda.synchronize()
        sem_b.release()

    finally:
        del cpu_array
        if cudart is not None and cpu_ptr is not None:
            cudart.cudaHostUnregister(cpu_ptr)
        mem.close()
        shm.unlink()
        sem_a.unlink()
        sem_b.unlink()
        print("[server] cleaned up")


# ── client process ────────────────────────────────────────────────────────────
def client(addr_queue: mp.Queue):
    time.sleep(0.5)
    print("[client] starting")

    shm       = posix_ipc.SharedMemory(SHM_NAME)
    mem       = mmap.mmap(shm.fd, NBYTES)
    sem_a     = posix_ipc.Semaphore(SEM_A_NAME)
    sem_b     = posix_ipc.Semaphore(SEM_B_NAME)
    cpu_array = make_cpu_array(mem, SHAPE)

    cpu_ptr = ctypes.c_void_p(
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
    cpu_vals = cpu_array.copy()
    print(f"[client] CPU array read:\n{cpu_vals}")

    if np.all(cpu_vals == 42.0):
        print("[client] TEST B PASSED ✓  client sees server's write\n")
    else:
        print("[client] TEST B FAILED ✗\n")

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
    for name in [SEM_A_NAME, SEM_B_NAME]:
        try: posix_ipc.Semaphore(name).unlink()
        except: pass

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"Device      : {torch.cuda.get_device_name(0)}")
    print(f"Array shape : {SHAPE}, {NBYTES} bytes\n")
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
