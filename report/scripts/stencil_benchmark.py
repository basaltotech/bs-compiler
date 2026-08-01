import torch
import time
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import get_energy_joules

def make_stencil_coeffs(dims: int, radius: int, order: int) -> List[float]:
    total = (2 * radius + 1) ** dims
    return [1.0 / total] * total

def make_stencil_fn(dims: int, radius: int, coeffs: List[float]):
    if dims == 1:
        def stencil(x):
            result = torch.zeros_like(x)
            idx = 0
            pad = (radius, radius)
            x_pad = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            for dx in range(-radius, radius + 1):
                shifted = torch.roll(x_pad, shifts=dx, dims=0)
                central = shifted[radius:radius + x.shape[0]]
                result += coeffs[idx] * central
                idx += 1
            return result
    elif dims == 2:
        def stencil(x):
            result = torch.zeros_like(x)
            idx = 0
            pad = (radius, radius, radius, radius)
            x_pad = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    shifted = torch.roll(x_pad, shifts=(dy, dx), dims=(0, 1))
                    central = shifted[radius:radius + x.shape[0], radius:radius + x.shape[1]]
                    result += coeffs[idx] * central
                    idx += 1
            return result
    elif dims == 3:
        def stencil(x):
            result = torch.zeros_like(x)
            idx = 0
            pad = (radius, radius, radius, radius, radius, radius)
            x_pad = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        shifted = torch.roll(x_pad, shifts=(dz, dy, dx), dims=(0, 1, 2))
                        central = shifted[radius:radius + x.shape[0],
                                         radius:radius + x.shape[1],
                                         radius:radius + x.shape[2]]
                        result += coeffs[idx] * central
                        idx += 1
            return result
    else:
        raise ValueError("dims must be 1, 2, or 3")
    return stencil

def benchmark_stencil(shape: Tuple[int, ...], dims: int, radius: int, order: int,
                      dtype: torch.dtype, device: str, backend: str,
                      num_iterations: int = 1000, warmup: int = 10) -> Dict[str, Any]:
    x = torch.randn(shape, dtype=dtype, device=device)
    coeffs = make_stencil_coeffs(dims, radius, order)
    stencil_fn = make_stencil_fn(dims, radius, coeffs)

    compiled = torch.compile(stencil_fn, backend=backend)

    for _ in range(warmup):
        _ = compiled(x)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = compiled(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start_time) / num_iterations * 1000

    eager_out = stencil_fn(x)
    compiled_out = compiled(x)
    if backend == "basalto":
        is_close = torch.allclose(compiled_out, eager_out, atol=1e-5, rtol=1e-5)
    else:
        is_close = True

    energy_joules = None
    if backend == "basalto":
        energy_joules = get_energy_joules()

    return {
        "backend": backend,
        "time_ms": elapsed,
        "correct": is_close,
        "energy_joules": energy_joules
    }

def run():
    device = "cuda"
    dtypes = [torch.float32]
    shapes_1d = [(4096,), (16384,), (65536,)]
    shapes_2d = [(128, 128), (256, 256), (512, 512)]
    shapes_3d = [(128, 128, 128), (256, 256, 256), (512, 512, 512)]
    orders = [2, 8, 12]
    results = {}
    for dims, shape_list in [(1, shapes_1d), (2, shapes_2d), (3, shapes_3d)]:
        for shape in shape_list:
            for order in orders:
                radius = order // 2
                key = f"{dims}D_{shape}_{order}"
                results[key] = {}
                for backend in ["basalto", "inductor"]:
                    try:
                        res = benchmark_stencil(shape, dims, radius, order,
                                                torch.float32, device, backend,
                                                num_iterations=1000, warmup=10)
                        results[key][backend] = res
                    except Exception as e:
                        results[key][backend] = {"error": str(e)}
    return results

if __name__ == "__main__":
    res = run()
    print(res)