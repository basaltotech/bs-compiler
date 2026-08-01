import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="report.md")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    lines = []
    lines.append("# Basalto Benchmark Report\n")
    lines.append(f"**Data:** {data['timestamp']}\n")
    lines.append("## Sistema\n")
    for k, v in data["system_info"].items():
        lines.append(f"- **{k}:** {v}")
    lines.append("\n")

    if "stencil" in data["benchmarks"]:
        lines.append("## Stencils\n")
        for name, results in data["benchmarks"]["stencil"].items():
            lines.append(f"### {name}\n")
            lines.append("| Backend | Tempo (ms) | Corrigido? | Energia (J) |\n")
            lines.append("|---------|------------|------------|-------------|\n")
            for backend, res in results.items():
                if "error" in res:
                    lines.append(f"| {backend} | Erro | - | - |\n")
                else:
                    time_ms = res.get("time_ms", "N/A")
                    correct = "✅" if res.get("correct", False) else "❌"
                    energy = f"{res.get('energy_joules', 'N/A'):.2f}" if res.get('energy_joules') else "N/A"
                    lines.append(f"| {backend} | {time_ms:.2f} | {correct} | {energy} |\n")
            lines.append("\n")

    if "matmul" in data["benchmarks"]:
        lines.append("## MatMul\n")
        for name, results in data["benchmarks"]["matmul"].items():
            lines.append(f"### {name}\n")
            lines.append("| Backend | Tempo (ms) | Corrigido? |\n")
            lines.append("|---------|------------|------------|\n")
            for backend, res in results.items():
                if "error" in res:
                    lines.append(f"| {backend} | Erro | - |\n")
                else:
                    time_ms = res.get("time_ms", "N/A")
                    correct = "✅" if res.get("correct", False) else "❌"
                    lines.append(f"| {backend} | {time_ms:.2f} | {correct} |\n")
            lines.append("\n")

    if "energy" in data["benchmarks"]:
        lines.append("## Energia\n")
        res = data["benchmarks"]["energy"]
        if "error" in res:
            lines.append(f"Erro: {res['error']}\n")
        else:
            lines.append(f"- **Energia consumida:** {res.get('energy_joules', 'N/A')} J\n")
            lines.append(f"- **Tempo:** {res.get('time_sec', 'N/A'):.2f} s\n")
            lines.append(f"- **Iterações:** {res.get('iterations', 'N/A')}\n")
            lines.append(f"- **Corrigido:** {'✅' if res.get('correct', False) else '❌'}\n")

    with open(args.output, "w") as f:
        f.writelines(lines)
    print(f"Relatório gerado: {args.output}")

if __name__ == "__main__":
    main()