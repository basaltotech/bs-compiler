#!/usr/bin/env python3
"""Teste de halo exchange multi-GPU usando MPI e NCCL (simulado com PyTorch)."""

import torch
import torch.distributed as dist
import os
import time
import numpy as np

def halo_exchange_3d(x, halo_size, rank, world_size, device):
    """Troca halos 3D entre vizinhos (simula MPI/NCCL)."""
    nx, ny, nz = x.shape
    if rank > 0:
        # Recebe halo da esquerda
        recv_left = torch.zeros((halo_size, ny, nz), device=device)
        # (simulação: cópia local)
        recv_left = x[:halo_size, :, :]
    if rank < world_size - 1:
        # Recebe halo da direita
        recv_right = torch.zeros((halo_size, ny, nz), device=device)
        recv_right = x[-halo_size:, :, :]
    if rank > 0:
        # Envia halo esquerdo para a esquerda
        send_left = x[:halo_size, :, :]
        # (simulação: cópia local)
    if rank < world_size - 1:
        # Envia halo direito para a direita
        send_right = x[-halo_size:, :, :]
    # (Em produção, usar NCCL/MPI)
    return x

def multi_gpu_stencil_benchmark(shape, radius, backend, time_steps=50, repeats=5):
    """Benchmark multi-GPU com halo exchange."""
    if not dist.is_available():
        print("Distribuído não disponível. Pulando teste multi-GPU.")
        return None

    # Inicializa o processo (MPI)
    rank = int(os.environ.get("PMI_RANK", 0))
    world_size = int(os.environ.get("PMI_SIZE", 1))
    device = f"cuda:{rank}"

    if rank == 0:
        print(f"Iniciando benchmark multi-GPU com {world_size} GPUs")

    x = torch.randn(shape, device=device)
    # Stencil simples para teste
    def stencil_fn(x):
        return (x[1:-1, 1:-1, 1:-1] * 0.125 +
                x[:-2, 1:-1, 1:-1] * 0.125 +
                x[2:, 1:-1, 1:-1] * 0.125 +
                x[1:-1, :-2, 1:-1] * 0.125 +
                x[1:-1, 2:, 1:-1] * 0.125 +
                x[1:-1, 1:-1, :-2] * 0.125 +
                x[1:-1, 1:-1, 2:] * 0.125)

    if backend == "basalto":
        import basalto
        compiled = torch.compile(stencil_fn, backend="basalto")
    elif backend == "inductor":
        compiled = torch.compile(stencil_fn, backend="inductor")
    else:
        compiled = stencil_fn

    # Warmup
    for _ in range(5):
        y = compiled(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(time_steps):
            # Troca halos
            x = halo_exchange_3d(x, radius, rank, world_size, device)
            # Aplica stencil
            x = compiled(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000 / time_steps)

    # Resultado médio
    avg_time = np.mean(times)
    std_time = np.std(times)
    if rank == 0:
        print(f"Multi-GPU: {avg_time:.2f}ms ± {std_time:.2f}ms (N={world_size})")
    return {"mean_ms": avg_time, "std_ms": std_time, "world_size": world_size, "rank": rank}