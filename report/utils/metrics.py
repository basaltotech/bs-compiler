"""Coleta de métricas do sistema e GPU."""

import subprocess
import json
import torch

def collect_system_info():
    info = {
        "hostname": subprocess.getoutput("hostname"),
        "gpu_model": subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").split("\n")[0],
        "cuda_version": subprocess.getoutput("nvcc --version | grep 'release' | awk '{print $6}'"),
        "torch_version": torch.__version__,
    }
    return info