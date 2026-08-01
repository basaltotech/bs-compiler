import torch
import time
from typing import Dict, Any

def matmul_fn(a, b):
    return torch.matmul(a, b)

def benchmark_matmul(m: int, n: int, k: int, dtype: torch.dtype,
                     device: str, backend: str, repeats: int = 20) -> Dict[str, Any]:
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    if backend == "eager":
        fn = matmul_fn
    else:
        fn = torch.compile(matmul_fn, backend=backend)

    for _ in range(5):
        _ = fn(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        _ = fn(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000

    eager_out = matmul_fn(a, b)
    compiled_out = fn(a, b)
    is_close = torch.allclose(compiled_out, eager_out, atol=1e-5, rtol=1e-5)

    return {
        "backend": backend,
        "time_ms": elapsed,
        "correct": is_close
    }

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
                try:
                    res = benchmark_matmul(m, n, k, dtype, device, backend)
                    results[key][backend] = res
                except Exception as e:
                    results[key][backend] = {"error": str(e)}
    return results

if __name__ == "__main__":
    res = run()
    print(res)