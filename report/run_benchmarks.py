#!/usr/bin/env python3
"""Orquestrador de benchmarks do Basalto."""

import argparse
import json
import time
import os
import sys
from pathlib import Path

# Adiciona scripts e utils ao path
sys.path.append(str(Path(__file__).parent / "scripts"))
sys.path.append(str(Path(__file__).parent / "utils"))

from scripts import stencil_benchmark, matmul_benchmark, energy_benchmark  # pyright: ignore[reportAttributeAccessIssue]
from utils.metrics import collect_system_info
from utils.report_utils import save_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/benchmark_results.json",
                        help="Arquivo de saída para os resultados")
    parser.add_argument("--no-stencil", action="store_true", help="Pular stencils")
    parser.add_argument("--no-matmul", action="store_true", help="Pular MatMul")
    parser.add_argument("--no-energy", action="store_true", help="Pular medição de energia")
    args = parser.parse_args()

    # Garante que o diretório de saída existe
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "system_info": collect_system_info(),
        "timestamp": time.time(),
        "benchmarks": {}
    }

    if not args.no_stencil:
        print("=== Executando benchmarks de stencil ===")
        results["benchmarks"]["stencil"] = stencil_benchmark.run()
        print("Stencil concluído.")

    if not args.no_matmul:
        print("=== Executando benchmarks de MatMul ===")
        results["benchmarks"]["matmul"] = matmul_benchmark.run()
        print("MatMul concluído.")

    if not args.no_energy:
        print("=== Executando benchmarks de energia ===")
        results["benchmarks"]["energy"] = energy_benchmark.run()
        print("Energia concluído.")

    # Salva resultados
    save_results(results, args.output)
    print(f"Resultados salvos em {args.output}")

if __name__ == "__main__":
    main()