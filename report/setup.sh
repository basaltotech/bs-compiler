#!/bin/bash
set -euo pipefail

echo "=== Basalto Benchmark Setup ==="
sudo apt-get update
sudo apt-get install -y build-essential curl git python3 python3-pip python3-venv

if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

if ! command -v maturin &> /dev/null; then
    pip3 install maturin
fi

# Clona e compila o Basalto (se não existir)
cd ..
if [ ! -d "bs-compiler" ]; then
    git clone https://github.com/basaltotech/bs-compiler.git
    cd bs-compiler
else
    cd bs-compiler
    git pull
fi

maturin build --release
pip3 install target/wheels/basalto-*.whl

# Dependências Python para benchmarks
pip3 install -r report/requirements.txt

echo "Setup concluído! Execute: python3 run_benchmarks.py"