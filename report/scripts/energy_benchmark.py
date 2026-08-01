import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import get_energy_joules

def stencil_3d(x):
    return (x[..., 1:-1, 1:-1, 1:-1] * 0.125 +
            x[..., :-2, 1:-1, 1:-1] * 0.125 +
            x[..., 2:, 1:-1, 1:-1] * 0.125 +
            x[..., 1:-1, :-2, 1:-1] * 0.125 +
            x[..., 1:-1, 2:, 1:-1] * 0.125 +
            x[..., 1:-1, 1:-1, :-2] * 0.125 +
            x[..., 1:-1, 1:-1, 2:] * 0.125)

def run():
    device = "cuda"
    shape = (128, 128, 128)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    compiled = torch.compile(stencil_3d, backend="basalto")
    for _ in range(5):
        _ = compiled(x)
    torch.cuda.synchronize()

    energy_before = get_energy_joules()
    start = time.perf_counter()
    for _ in range(100):
        _ = compiled(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    energy_after = get_energy_joules()

    if energy_before is not None and energy_after is not None:
        energy_used = energy_after - energy_before
    else:
        energy_used = None

    eager_out = stencil_3d(x)
    compiled_out = compiled(x)
    is_close = torch.allclose(compiled_out, eager_out, atol=1e-5, rtol=1e-5)

    return {
        "energy_joules": energy_used,
        "time_sec": elapsed,
        "correct": is_close,
        "iterations": 100
    }

if __name__ == "__main__":
    res = run()
    print(res)