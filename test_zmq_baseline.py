"""
test_zmq_baseline.py
--------------------
ZMQ REQ-REP transport benchmark — mirrors the GR00T PolicyServer/PolicyClient
wire format exactly (msgpack envelope + per-array np.save() encoding).

No GR00T or model dependencies needed; runs on a plain numpy + zmq + msgpack
install. The server returns dummy action tensors of the correct shape.

Observation format: OXE_WIDOWX / Gr00tSimPolicyWrapper flat keys
  video.image_0                       (B, 1, 256, 256, 3) uint8
  state.{x,y,z,roll,pitch,yaw,pad,gripper}  (B, 1, 1) float32  — 8 keys
  annotation.human.action.task_description  list[str] length B

Action format (returned by server):
  action.{x,y,z,roll,pitch,yaw,gripper}     (B, 8, 1) float32  — 7 keys

Two rounds:
  B=1  simulates a single real robot
  B=5  simulates sim eval with n_envs=5

Run:
  python test_zmq_baseline.py
  python -m cProfile -s cumtime test_zmq_baseline.py   # CPU profiling
  nsys profile --trace=nvtx python test_zmq_baseline.py
"""

import io
import multiprocessing as mp
import time

import msgpack
import numpy as np

# ── config ─────────────────────────────────────────────────────────────────────
ZMQ_ADDR       = "tcp://127.0.0.1:5556"
N_ITERS        = 200
IMG_H, IMG_W   = 256, 256
ACTION_HORIZON = 8
STATE_KEYS     = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_KEYS    = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANG_INSTR     = "Pick up the eggplant and put it in the basket."
# ───────────────────────────────────────────────────────────────────────────────


# ── serializer ─────────────────────────────────────────────────────────────────
# Exact copy of gr00t/policy/server_client.py MsgSerializer.
# msgpack 1.x defaults to raw=False (string keys), which is what GR00T uses.
class MsgSerializer:
    @staticmethod
    def to_bytes(data) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode, raw=False)

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Unserializable type: {type(obj)}")

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


# ── helpers ────────────────────────────────────────────────────────────────────
def build_obs(B: int) -> dict:
    """Synthetic WidowX observation in Gr00tSimPolicyWrapper flat-key format."""
    obs = {
        "video.image_0": np.random.randint(0, 256, (B, 1, IMG_H, IMG_W, 3), dtype=np.uint8),
    }
    for k in STATE_KEYS:
        obs[f"state.{k}"] = np.random.randn(B, 1, 1).astype(np.float32)
    obs["annotation.human.action.task_description"] = [LANG_INSTR] * B
    return obs


def build_dummy_action(B: int) -> tuple:
    actions = {
        f"action.{k}": np.zeros((B, ACTION_HORIZON, 1), dtype=np.float32)
        for k in ACTION_KEYS
    }
    return (actions, {})


# ── server process ─────────────────────────────────────────────────────────────
def server(ready: mp.Event):
    import zmq
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(ZMQ_ADDR)
    ready.set()

    while True:
        raw = sock.recv()
        req = MsgSerializer.from_bytes(raw)

        if req.get("endpoint") == "kill":
            sock.send(MsgSerializer.to_bytes({"status": "ok"}))
            break

        # Infer batch size from image array shape
        obs = req["data"]["observation"]
        B   = obs["video.image_0"].shape[0]
        sock.send(MsgSerializer.to_bytes(build_dummy_action(B)))

    sock.close()
    ctx.term()


# ── client ─────────────────────────────────────────────────────────────────────
def run_client(B: int) -> list:
    import zmq
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 10_000)
    sock.connect(ZMQ_ADDR)

    obs     = build_obs(B)
    request = {"endpoint": "get_action", "data": {"observation": obs, "options": None}}

    # warm-up (first call has JIT / import overhead)
    for _ in range(3):
        sock.send(MsgSerializer.to_bytes(request))
        sock.recv()

    records = []
    for _ in range(N_ITERS):
        t0  = time.perf_counter()
        msg = MsgSerializer.to_bytes(request)
        t1  = time.perf_counter()
        sock.send(msg)
        raw = sock.recv()
        t2  = time.perf_counter()
        MsgSerializer.from_bytes(raw)
        t3  = time.perf_counter()

        records.append({
            "total_ms":   (t3 - t0) * 1e3,
            "ser_ms":     (t1 - t0) * 1e3,
            "socket_ms":  (t2 - t1) * 1e3,
            "deser_ms":   (t3 - t2) * 1e3,
            "req_bytes":  len(msg),
            "resp_bytes": len(raw),
        })

    sock.send(MsgSerializer.to_bytes({"endpoint": "kill"}))
    sock.recv()
    sock.close()
    ctx.term()
    return records


# ── report ─────────────────────────────────────────────────────────────────────
def report(title: str, records: list):
    a       = {k: np.array([r[k] for r in records]) for k in records[0]}
    req_kb  = records[0]["req_bytes"]  / 1024
    resp_kb = records[0]["resp_bytes"] / 1024

    print(f"\n── {title} ──")
    print(f"  Payload : req = {req_kb:.1f} KB   resp = {resp_kb:.2f} KB")
    hdr = f"  {'component':<22} {'mean':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}  ms"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, key in [
        ("total",          "total_ms"),
        ("  serialize",    "ser_ms"),
        ("  socket+copy",  "socket_ms"),
        ("  deserialize",  "deser_ms"),
    ]:
        v = a[key]
        print(f"  {label:<22} {np.mean(v):>7.3f} {np.percentile(v,50):>7.3f} "
              f"{np.percentile(v,95):>7.3f} {np.percentile(v,99):>7.3f} {np.max(v):>7.3f}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn")

    print("ZMQ Baseline — GR00T WidowX wire format")
    print(f"  Image    : {IMG_H}×{IMG_W}×3 uint8  (~{IMG_H*IMG_W*3/1024:.0f} KB per env)")
    print(f"  State    : {len(STATE_KEYS)} keys × (B, 1, 1) float32")
    print(f"  Action   : {len(ACTION_KEYS)} keys × (B, {ACTION_HORIZON}, 1) float32")
    print(f"  Iters    : {N_ITERS}  (+3 warm-up)")
    print(f"  Address  : {ZMQ_ADDR}")
    print("=" * 55)

    for B in [1, 5]:
        label = "B=1  real robot" if B == 1 else "B=5  sim eval (n_envs=5)"
        ready = mp.Event()
        p = mp.Process(target=server, args=(ready,))
        p.start()
        ready.wait()

        records = run_client(B)
        p.join()
        report(label, records)

    print("\nDone.")
