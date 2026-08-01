"""Benchmark de MatMul denso e fundido."""

import torch
import time

def matmul_fn(a, b):
    return torch.matmul(a, b)

def benchmark_matmul(m, n, k, dtype, backend, device, repeats=20):
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    compiled = torch.compile(matmul_fn, backend=backend)
    for _ in range(3):
        compiled(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        compiled(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000

def run():
    device = "cuda"
    dtypes = [torch.float32, torch.float64]
    shapes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    results = {}
    for m, n, k in shapes:
        for dtype in dtypes:
            key = f"{m}x{n}x{k}_{dtype}".replace("torch.", "")
            results[key] = {}
            for backend in ["basalto", "inductor", "eager"]:
                if backend == "eager":
                    fn = matmul_fn
                else:
                    fn = torch.compile(matmul_fn, backend=backend)
                try:
                    t = benchmark_matmul(m, n, k, dtype, backend, device)
                    results[key][backend] = t
                except Exception as e:
                    results[key][backend] = None
    return results