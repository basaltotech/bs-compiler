"""Gera relatório em Markdown a partir dos resultados JSON."""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Arquivo JSON de resultados")
    parser.add_argument("--output", default="report.md", help="Arquivo de saída (Markdown)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    output = []
    output.append("# Basalto Benchmark Report\n")
    output.append(f"**Data:** {data['timestamp']}\n")
    output.append("## Sistema\n")
    for k, v in data["system_info"].items():
        output.append(f"- **{k}:** {v}")
    output.append("\n")

    if "stencil" in data["benchmarks"]:
        output.append("## Stencils\n")
        for name, results in data["benchmarks"]["stencil"].items():
            output.append(f"### {name}\n")
            output.append("| Backend | Tempo (ms) | Speedup vs Inductor |\n")
            output.append("|---------|------------|---------------------|\n")
            basalto_time = results.get("basalto")
            inductor_time = results.get("inductor")
            if basalto_time and inductor_time:
                speedup = inductor_time / basalto_time if basalto_time > 0 else 0
                output.append(f"| Basalto | {basalto_time:.2f} | {speedup:.2f}x |\n")
                output.append(f"| Inductor | {inductor_time:.2f} | 1.00x |\n")
            else:
                output.append("| Basalto | N/A | N/A |\n")
                output.append("| Inductor | N/A | N/A |\n")
            output.append("\n")

    if "matmul" in data["benchmarks"]:
        output.append("## MatMul\n")
        for name, results in data["benchmarks"]["matmul"].items():
            output.append(f"### {name}\n")
            output.append("| Backend | Tempo (ms) | Speedup vs Eager |\n")
            output.append("|---------|------------|------------------|\n")
            for backend in ["basalto", "inductor", "eager"]:
                t = results.get(backend)
                if t:
                    eager_t = results.get("eager", 1)
                    speedup = eager_t / t if t > 0 else 0
                    output.append(f"| {backend.capitalize()} | {t:.2f} | {speedup:.2f}x |\n")
                else:
                    output.append(f"| {backend.capitalize()} | N/A | N/A |\n")
            output.append("\n")

    if "energy" in data["benchmarks"]:
        output.append("## Energia\n")
        output.append("Consumo registrado pelo Basalto: verifique os logs em `/var/log/basalto/basalto.log`.\n")

    with open(args.output, "w") as f:
        f.writelines(output)
    print(f"Relatório gerado: {args.output}")

if __name__ == "__main__":
    main()