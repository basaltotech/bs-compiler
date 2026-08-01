#!/bin/bash
set -euo pipefail

# ============================================================
# Basalto Enterprise Suite - Instalador Completo (MVP)
# ============================================================
# Uso: sudo ./install.sh [--uninstall] [--version X.Y.Z] [--env-file .env]
# ============================================================

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'       
NC='\033[0m'

# ============================================================
# Variáveis padrão (podem ser sobrescritas pelo .env)
# ============================================================
BASALTO_VERSION="${BASALTO_VERSION:-latest}"
BASALTO_INSTALL_DIR="${BASALTO_INSTALL_DIR:-/usr/local/bin}"
BASALTO_CONFIG_DIR="${BASALTO_CONFIG_DIR:-/etc/basalto}"
BASALTO_LOG_DIR="${BASALTO_LOG_DIR:-/var/log/basalto}"
BASALTO_CACHE_DIR="${BASALTO_CACHE_DIR:-/var/cache/basalto}"
BASALTO_USER="${BASALTO_USER:-$SUDO_USER}"
BASALTO_GROUP="${BASALTO_GROUP:-$BASALTO_USER}"
BASALTO_REPO="${BASALTO_REPO:-https://github.com/basaltotech/bs-compiler/releases/download}"
BASALTO_BINARY="basalto-installer"
BASALTO_CHECKSUM_FILE="checksums.txt"
BASALTO_WHEEL_PATTERN="basalto-*.whl"
BASALTO_ENV_FILE="${BASALTO_ENV_FILE:-.env}"
BASALTO_INSTALL_ENV="${BASALTO_INSTALL_ENV:-system}"  # system ou venv
BASALTO_PYTHON="${BASALTO_PYTHON:-python3}"

# ============================================================
# Funções de log
# ============================================================
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success(){ echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ============================================================
# Função para carregar variáveis do .env (se existir)
# ============================================================
load_env() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        log_info "Carregando variáveis de $env_file"
        set -a
        source "$env_file"
        set +a
    fi
}

# ============================================================
# 1. Verificações iniciais
# ============================================================
if [ "$EUID" -ne 0 ]; then
    log_error "Este script deve ser executado como root (sudo)."
    exit 1
fi

if ! command -v "$BASALTO_PYTHON" &>/dev/null; then
    log_error "Python 3 não encontrado. Instale python3."
    exit 1
fi

if ! command -v pip &>/dev/null && ! command -v pip3 &>/dev/null; then
    log_error "pip não encontrado. Instale pip (apt install python3-pip ou equivalente)."
    exit 1
fi

# ============================================================
# 2. Processamento de argumentos
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --uninstall)
            log_info "Desinstalando Basalto..."
            # Remove binário e diretórios
            if [ -f "$BASALTO_INSTALL_DIR/$BASALTO_BINARY" ]; then
                rm -f "$BASALTO_INSTALL_DIR/$BASALTO_BINARY"
                log_success "Binário removido: $BASALTO_INSTALL_DIR/$BASALTO_BINARY"
            fi
            for dir in "$BASALTO_CONFIG_DIR" "$BASALTO_LOG_DIR" "$BASALTO_CACHE_DIR"; do
                if [ -d "$dir" ]; then
                    rm -rf "$dir"
                    log_success "Removido: $dir"
                fi
            done
            # Remove pacote Python
            if command -v pip &>/dev/null; then
                pip uninstall -y basalto 2>/dev/null && log_success "Pacote basalto removido do pip." || log_warn "Pacote basalto não encontrado no pip."
            fi
            if command -v pip3 &>/dev/null; then
                pip3 uninstall -y basalto 2>/dev/null && log_success "Pacote basalto removido do pip3." || true
            fi
            log_success "Basalto desinstalado com sucesso."
            exit 0
            ;;
        --version)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                BASALTO_VERSION="$2"
                shift
            else
                log_error "Versão inválida. Use: --version X.Y.Z"
                exit 1
            fi
            ;;
        --env-file)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                BASALTO_ENV_FILE="$2"
                shift
            else
                log_error "Arquivo .env inválido."
                exit 1
            fi
            ;;
        --venv)
            BASALTO_INSTALL_ENV="venv"
            log_info "Instalação em ambiente virtual (venv) será criada em ./basalto-venv"
            ;;
        --help|-h)
            cat <<EOF
Uso: sudo ./install.sh [OPÇÕES]

Opções:
  --uninstall              Remove o Basalto do sistema
  --version X.Y.Z          Instala uma versão específica (padrão: latest)
  --env-file /path/.env    Carrega variáveis de ambiente do arquivo (padrão: .env)
  --venv                   Instala em um ambiente virtual (./basalto-venv) em vez de system
  --help, -h               Exibe esta mensagem
EOF
            exit 0
            ;;
        *)
            log_error "Opção desconhecida: $1"
            exit 1
            ;;
    esac
    shift
done

# ============================================================
# 3. Carregar .env (se existir)
# ============================================================
load_env "$BASALTO_ENV_FILE"

# ============================================================
# 4. Detecção de arquitetura e sistema
# ============================================================
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$ARCH" in
    x86_64)  ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    *) log_error "Arquitetura não suportada: $ARCH"; exit 1 ;;
esac
if [ "$OS" != "linux" ]; then
    log_error "Sistema operacional não suportado: $OS"
    exit 1
fi
log_info "Detectado: $OS / $ARCH"

# ============================================================
# 5. Obter versão (se 'latest')
# ============================================================
if [ "$BASALTO_VERSION" = "latest" ]; then
    if command -v curl &>/dev/null; then
        LATEST_URL="https://api.github.com/repos/basaltotech/bs-compiler/releases/latest"
        BASALTO_VERSION=$(curl -s "$LATEST_URL" | grep -Po '"tag_name": "\K.*?(?=")')
        if [ -z "$BASALTO_VERSION" ]; then
            log_error "Falha ao obter a última versão do GitHub."
            exit 1
        fi
        log_info "Última versão encontrada: $BASALTO_VERSION"
    else
        log_error "curl não encontrado. Instale curl ou especifique uma versão com --version."
        exit 1
    fi
fi

# ============================================================
# 6. Montar URLs e baixar artefatos
# ============================================================
BINARY_URL="$BASALTO_REPO/$BASALTO_VERSION/${BASALTO_BINARY}_${OS}_${ARCH}"
CHECKSUM_URL="$BASALTO_REPO/$BASALTO_VERSION/$BASALTO_CHECKSUM_FILE"
WHEEL_URL="$BASALTO_REPO/$BASALTO_VERSION/${BASALTO_WHEEL_PATTERN}"

TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

log_info "Baixando artefatos da versão $BASALTO_VERSION..."

# 6.1 Baixar binário do instalador
log_info "Baixando $BASALTO_BINARY..."
if ! curl -L --fail --progress-bar -o "$TMP_DIR/$BASALTO_BINARY" "$BINARY_URL"; then
    log_error "Falha ao baixar o binário. Verifique a URL e sua conexão."
    exit 1
fi
chmod +x "$TMP_DIR/$BASALTO_BINARY"

# 6.2 Baixar checksums (se disponível)
if command -v sha256sum &>/dev/null && curl -s --fail -o "$TMP_DIR/checksums.txt" "$CHECKSUM_URL"; then
    log_info "Verificando integridade do binário..."
    EXPECTED_CHECKSUM=$(grep "$BASALTO_BINARY" "$TMP_DIR/checksums.txt" | awk '{print $1}')
    if [ -n "$EXPECTED_CHECKSUM" ]; then
        ACTUAL_CHECKSUM=$(sha256sum "$TMP_DIR/$BASALTO_BINARY" | awk '{print $1}')
        if [ "$EXPECTED_CHECKSUM" != "$ACTUAL_CHECKSUM" ]; then
            log_error "Checksum inválido! Esperado: $EXPECTED_CHECKSUM, Obtido: $ACTUAL_CHECKSUM"
            exit 1
        fi
        log_success "Checksum do binário verificado."
    else
        log_warn "Checksum não encontrado para o binário. Pulando verificação."
    fi
else
    log_warn "sha256sum ou checksum não disponível. Pulando verificação."
fi

# 6.3 Baixar a wheel (se existir)
WHEEL_FILE=""
if curl -s --fail -L -o "$TMP_DIR/wheel_download" "$WHEEL_URL" 2>/dev/null; then
    # Se o download funcionou, procura o arquivo .whl
    WHEEL_FILE=$(find "$TMP_DIR" -maxdepth 1 -name "*.whl" | head -1)
    if [ -n "$WHEEL_FILE" ]; then
        log_success "Wheel baixada: $(basename "$WHEEL_FILE")"
    fi
else
    log_warn "Wheel pré-compilada não encontrada para esta versão. Tentando instalar a partir do código fonte..."
fi

# ============================================================
# 7. Instalar via pip (wheel ou fonte)
# ============================================================
if [ "$BASALTO_INSTALL_ENV" = "venv" ]; then
    VENV_DIR="${BASALTO_VENV_DIR:-./basalto-venv}"
    log_info "Criando ambiente virtual em $VENV_DIR"
    "$BASALTO_PYTHON" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    PIP_CMD="$VENV_DIR/bin/pip"
else
    PIP_CMD="pip3"
fi

if [ -n "$WHEEL_FILE" ]; then
    log_info "Instalando wheel: $WHEEL_FILE"
    if ! "$PIP_CMD" install "$WHEEL_FILE"; then
        log_error "Falha ao instalar a wheel."
        exit 1
    fi
    log_success "Wheel instalada com sucesso."
else
    log_info "Instalando diretamente do GitHub (pip install git+...)"
    if ! "$PIP_CMD" install git+https://github.com/basaltotech/bs-compiler.git; then
        log_error "Falha ao instalar do GitHub. Certifique-se de que o Rust e o maturin estão instalados."
        exit 1
    fi
    log_success "Instalação do GitHub concluída."
fi

# ============================================================
# 8. Instalação do binário e diretórios
# ============================================================
log_info "Instalando binário do instalador em $BASALTO_INSTALL_DIR..."
mkdir -p "$BASALTO_INSTALL_DIR"
cp "$TMP_DIR/$BASALTO_BINARY" "$BASALTO_INSTALL_DIR/"
chmod 755 "$BASALTO_INSTALL_DIR/$BASALTO_BINARY"

log_info "Criando diretórios..."
mkdir -p "$BASALTO_CONFIG_DIR" "$BASALTO_LOG_DIR" "$BASALTO_CACHE_DIR"

if [ -n "$BASALTO_USER" ] && id "$BASALTO_USER" &>/dev/null; then
    chown -R "$BASALTO_USER":"$BASALTO_GROUP" "$BASALTO_CONFIG_DIR" "$BASALTO_LOG_DIR" "$BASALTO_CACHE_DIR" 2>/dev/null || true
    chmod 755 "$BASALTO_CONFIG_DIR" "$BASALTO_LOG_DIR" "$BASALTO_CACHE_DIR"
fi

# ============================================================
# 9. Criar arquivo de configuração (se não existir)
# ============================================================
CONFIG_FILE="$BASALTO_CONFIG_DIR/config.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    log_info "Criando arquivo de configuração padrão..."
    cat > "$CONFIG_FILE" <<EOF
# Basalto Configuration (gerado automaticamente)

[node]
node_id = "$(hostname)"
cluster_name = "default"

[cache]
cache_dir = "$BASALTO_CACHE_DIR"
max_size_mb = 10240

[logging]
level = "info"
file = "$BASALTO_LOG_DIR/basalto.log"

[redis]
enabled = false
url = "redis://localhost:6379"

[audit]
enabled = false

[telemetry]
energy_source = "auto"
bmc_ip = ""
bmc_user = ""
bmc_password = ""
EOF
    log_success "Arquivo de configuração criado em $CONFIG_FILE"
fi

# ============================================================
# 10. Verificação de dependências (apenas avisos)
# ============================================================
log_info "Verificando dependências..."
command -v nvidia-smi &>/dev/null && log_success "NVIDIA drivers detectados." || log_warn "nvidia-smi não encontrado."
command -v mpirun &>/dev/null && log_success "MPI detectado." || log_warn "MPI não encontrado."
command -v redis-cli &>/dev/null && log_success "Redis detectado." || log_warn "Redis não encontrado."

# ============================================================
# 11. Execução do instalador Rust (basalto-installer)
# ============================================================
log_info "Executando o instalador Rust para configurar o sistema..."
if ! "$BASALTO_INSTALL_DIR/$BASALTO_BINARY" 2>&1 | tee "$BASALTO_LOG_DIR/install.log"; then
    log_error "Falha na execução do basalto-installer. Verifique $BASALTO_LOG_DIR/install.log"
    exit 1
fi

# ============================================================
# 12. Finalização
# ============================================================
log_success "Basalto instalado e configurado com sucesso!"
log_info "Versão instalada: $BASALTO_VERSION"
log_info "Binário: $BASALTO_INSTALL_DIR/$BASALTO_BINARY"
log_info "Configuração: $CONFIG_FILE"
log_info "Logs: $BASALTO_LOG_DIR/install.log"

if [ "$BASALTO_INSTALL_ENV" = "venv" ]; then
    log_info "Ambiente virtual: $VENV_DIR (ative com 'source $VENV_DIR/bin/activate')"
fi

cat <<EOF

Para testar a instalação, execute em Python:

    import basalto
    import torch

    # Registra o backend
    torch.compile(..., backend="basalto")

Para configurar o cache Redis, edite: $CONFIG_FILE
Para habilitar auditoria, defina BASALTO_AUDIT_ENABLED=true

Após a instalação, recarregue as GPUs (opcional):
    sudo udevadm trigger --type=subsystems --action=add /sys/class/nvidia

EOF