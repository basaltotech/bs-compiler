
---

### 2. `report/setup.sh`

```bash
#!/bin/bash
set -euo pipefail

echo "=== Basalto Benchmark Setup ==="

# Instala dependências do sistema
sudo apt-get update
sudo apt-get install -y build-essential curl git python3 python3-pip python3-venv

# Instala Rust
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Instala maturin
if ! command -v maturin &> /dev/null; then
    pip3 install maturin
fi

# Instala o Basalto (modo desenvolvimento)
cd ..
if [ -d "basalto" ]; then
    echo "Basalto já está presente, atualizando..."
    cd basalto
    git pull
else
    echo "Clonando Basalto..."
    git clone https://github.com/basaltotech/bs-compiler.git basalto
    cd basalto
fi

# Build e instala a wheel
maturin build --release
pip3 install target/wheels/basalto-*.whl

# Instala dependências Python para benchmarks
pip3 install -r report/requirements.txt

echo "Setup concluído!"
echo "Agora execute: python run_benchmarks.py"