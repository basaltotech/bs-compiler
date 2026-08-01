import subprocess
import torch
import os

def collect_system_info():
    info = {
        "hostname": subprocess.getoutput("hostname"),
        "gpu_model": subprocess.getoutput("nvidia-smi --query-gpu=name --format=csv,noheader").split("\n")[0],
        "cuda_version": subprocess.getoutput("nvcc --version | grep 'release' | awk '{print $6}'"),
        "torch_version": torch.__version__,
    }
    return info

def get_energy_joules():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return energy_mj / 1000.0
    except:
        try:
            out = subprocess.check_output(
                "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
                shell=True, text=True
            )
            return None
        except:
            return None