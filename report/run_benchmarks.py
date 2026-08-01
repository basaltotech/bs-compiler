#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from scripts import stencil_benchmark
from scripts import matmul_benchmark
from scripts import energy_benchmark
from utils.metrics import collect_system_info
from utils.report_utils import save_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/benchmark_results.json")
    parser.add_argument("--no-stencil", action="store_true")
    parser.add_argument("--no-matmul", action="store_true")
    parser.add_argument("--no-energy", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = {
        "system_info": collect_system_info(),
        "timestamp": time.time(),
        "benchmarks": {}
    }

    if not args.no_stencil:
        print("=== Executando stencils ===")
        results["benchmarks"]["stencil"] = stencil_benchmark.run()
        print("Stencils concluídos.")

    if not args.no_matmul:
        print("=== Executando MatMul ===")
        results["benchmarks"]["matmul"] = matmul_benchmark.run()
        print("MatMul concluído.")

    if not args.no_energy:
        print("=== Executando teste de energia ===")
        results["benchmarks"]["energy"] = energy_benchmark.run()
        print("Energia concluído.")

    save_results(results, args.output)
    print(f"Resultados salvos em {args.output}")

if __name__ == "__main__":
    main()