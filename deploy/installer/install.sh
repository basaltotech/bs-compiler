#!/bin/bash
set -e
if [ "$EUID" -ne 0 ]; then
    echo "Execute com sudo: sudo ./install.sh"
    exit 1
fi

echo "Instalando Basalto..."
# Exemplo: baixa binário, executa instalador Rust
/usr/local/bin/basalto-installer
echo "Concluído."