"""Funções auxiliares para salvar e gerar relatórios."""

import json

def save_results(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)