"""Benchmark de stencils 1D, 2D, 3D com Basalto vs Inductor."""

import torch
import time
import numpy as np

def stencil_1d(x):
    return (x[..., :-2] + x[..., 1:-1] + x[..., 2:]) / 3.0

def stencil_2d(x):
    # média dos vizinhos (simples)
    return (x[..., 1:-1, 1:-1] * 0.2 +
            x[..., :-2, 1:-1] * 0.2 +
            x[..., 2:, 1:-1] * 0.2 +
            x[..., 1:-1, :-2] * 0.2 +
            x[..., 1:-1, 2:] * 0.2)

def stencil_3d(x):
    # média dos vizinhos em 3D (simples)
    return (x[..., 1:-1, 1:-1, 1:-1] * 0.125 +
            x[..., :-2, 1:-1, 1:-1] * 0.125 +
            x[..., 2:, 1:-1, 1:-1] * 0.125 +
            x[..., 1:-1, :-2, 1:-1] * 0.125 +
            x[..., 1:-1, 2:, 1:-1] * 0.125 +
            x[..., 1:-1, 1:-1, :-2] * 0.125 +
            x[..., 1:-1, 1:-1, 2:] * 0.125)

def benchmark_stencil(stencil_fn, shape, dtype, backend, device, repeats=20):
    x = torch.randn(shape, dtype=dtype, device=device)
    compiled = torch.compile(stencil_fn, backend=backend)
    # warmup
    for _ in range(3):
        compiled(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        compiled(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000  # ms
    return elapsed

def run():
    device = "cuda"
    dtypes = [torch.float32, torch.float64]
    shapes = {
        "1D": [(1024,), (4096,), (16384,)],
        "2D": [(128, 128), (256, 256), (512, 512)],
        "3D": [(32, 32, 32), (64, 64, 64)]
    }
    results = {}
    for dim, shape_list in shapes.items():
        fn = globals()[f"stencil_{dim.lower()}"]
        for dtype in dtypes:
            for shape in shape_list:
                key = f"{dim}_{shape}_{dtype}".replace("torch.", "")
                results[key] = {}
                for backend in ["basalto", "inductor"]:
                    try:
                        t = benchmark_stencil(fn, shape, dtype, backend, device)
                        results[key][backend] = t
                    except Exception as e:
                        results[key][backend] = None
    return results