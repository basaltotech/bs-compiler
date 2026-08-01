## 📁 Raiz do Projeto

| Arquivo/Diretório | Descrição |
| :--- | :--- |
| `Cargo.toml` | Workspace Rust (todos os crates) |
| `Cargo.lock` | Lockfile das dependências |
| `pyproject.toml` | Configuração do maturin (bindings Python) |
| `.env.example` | Exemplo de variáveis de ambiente (REDIS_CACHE_URL, LOG_LEVEL) |
| `.gitignore` | Arquivos ignorados pelo Git |
| `README.md` | Documentação inicial |
| `LICENSE` | Licença do projeto |

---

## 📁 `crates/` — Núcleo em Rust

### `basalto-common/` — Utilitários compartilhados

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/config.rs` | Leitura de `.env` |
| `src/error.rs` | Tipos de erro unificados |
| `src/hardware.rs` | Detecção dinâmica via APIs (CUDA/HIP/ZE) |
| `src/permissions.rs` | Verificação de CAP_SYS_NICE e device nodes (sem root) |
| `src/telemetry.rs` | Helpers para envio assíncrono de métricas (base para o COUN) |
| `src/hasher.rs` | (Nota: também presente em basalto-core) |

### `basalto-core/` — Núcleo do compilador (FLIR + Hasher)

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/flir_builder.rs` | Geração de FLIR a partir da AST Python |
| `src/hasher.rs` | SHA-256 com (estrutura, tipos, shape, vendor, arch, driver_version) |
| `src/llvm/` | Módulo LLVM |
| `src/llvm/mod.rs` | Ponto de entrada do módulo LLVM |
| `src/llvm/builder.rs` | Builder para IR LLVM |
| `src/llvm/parser.rs` | Parser para IR LLVM |
| `src/llvm/types.rs` | Definição de tipos LLVM |

> **Observação:** Os diretórios `ir/`, `analysis/`, `transforms/` e `dialect/` são mencionados na estrutura, mas não aparecem na listagem do `src/` — podem estar vazios ou ainda não commitados.

### `basalto-gems/` — Stride View (reorganização de memória sem cópia)

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/stride_view.rs` | Reinterpretação de layout de tensores |

### `basalto-target-nvidia/` — Codegen FLIR → PTX

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/codegen.rs` | Gera PTX via LLVM |
| `src/runtime.rs` | Chamadas CUDA para validação |

### `basalto-target-amd/` — Codegen FLIR → HSACO

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/codegen.rs` | Gera HSACO |
| `src/runtime.rs` | Runtime para AMD |

### `basalto-target-intel/` — Codegen FLIR → SPIR-V

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/codegen.rs` | Gera SPIR-V |
| `src/runtime.rs` | Runtime para Intel |

### `siliconforge-jit/` — Autocalibração contínua em tempo real

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/profiler.rs` | Roda em tokio::spawn (task assíncrona) — recalibra blocos matemáticos em background |

### `basalto-tree/` — Interceptor principal (registrado no torch.compile)

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Depende de basalto-core, gems, targets, common |
| `src/lib.rs` | Ponto de entrada |
| `src/interceptor.rs` | Fluxo: raw_call → gems::stride_view() → core::hash() → cache_lookup → (compile/reuse) → executor |
| `src/local_cache.rs` | Cache L1 (in-memory + disco local) |
| `src/cluster_cache.rs` | Cache L2 (Redis — somente binário). Consulta/escrita oportunista, não bloqueante |
| `src/executor.rs` | Dispara kernel na GPU local + NOTIFICA siliconforge-jit (assíncrono) |

### `energy-telemetry/` — Medição de kWh e correlação (COUN)

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Manifesto do crate |
| `src/lib.rs` | Ponto de entrada |
| `src/reader.rs` | Leitura de sensores (IPMI, Redfish, API proprietária) |
| `src/correlator.rs` | Amarra (hash, timestamp_início, timestamp_fim, kWh_delta) por execução |

### `basalto-installer/` — Binário Rust para instalação segura

| Arquivo | Descrição |
| :--- | :--- |
| `Cargo.toml` | Reutiliza basalto-common (hardware + permissions) |
| `src/main.rs` | Detecta hardware, negocia permissões mínimas, gera configuração |

---

## 📁 `python/` — Bindings PyO3 (expõe o Basalto Tree para o PyTorch)

| Arquivo/Diretório | Descrição |
| :--- | :--- |
| `Cargo.toml` | Depende de basalto-tree |
| `pyproject.toml` | Configuração do pacote Python |
| `src/` | Código fonte da extensão Rust (PyO3) |
| `basalto/` | Pacote Python |
| `basalto/__init__.py` | Importa o interceptor do Rust |
| `basalto/_rust.pyi` | Stubs para tipagem |
| `basalto/compiler.py` | Wrapper que registra o backend no torch.compile() |
| `basalto/lib.rs` | Código Rust da extensão (PyO3) |
| `basalto/ops/` | Autotune e heurísticas (mantido em Python) |
| `basalto/ops/__init__.py` | Inicialização do módulo ops |
| `basalto/ops/matmul.py` | Kernel de multiplicação de matrizes |
| `basalto/ops/attention.py` | Kernel de atenção |
| `tests/unit/` | Testes unitários |
| `tests/integration/` | Testes com GPU real (local) |

---

## 📁 `deploy/` — Implantação

| Arquivo/Diretório | Descrição |
| :--- | :--- |
| `installer/install.sh` | Bootstrap fino (bash): baixa o binário Rust, checa SO, executa o basalto-installer |
| `redis/redis.conf` | (Opcional) Configuração do cache compartilhado |

---

## 📁 `scripts/` — Utilitários para desenvolvimento

| Arquivo | Descrição |
| :--- | :--- |
| `dev-setup.sh` | Script de setup para desenvolvimento |
| `run-tests.sh` | Script para executar testes |

---

## 📁 `.github/` — CI/CD

| Arquivo | Descrição |
| :--- | :--- |
| `workflows/ci.yml` | Lint, testes unitários (sem GPU), build wheels |
| `workflows/integration.yml` | Testes com GPU real (self-hosted) |

---

## 📁 `docs/` — Documentação

| Arquivo | Descrição |
| :--- | :--- |
| `architecture.md` | Desenho arquitetural final |
| `integration.md` | Como registrar no PyTorch |
| `cache_protocol.md` | Especificação da chave do Redis (hash + arch + driver) |
| `energy_telemetry.md` | Como medir kWh e faturar em COUN |

---

## ✅ Resumo

O repositório contém **todos os crates Rust** necessários para o compilador, **bindings PyO3** para integração com Python/PyTorch, **scripts de deploy**, **CI/CD** e **documentação técnica**. A estrutura segue exatamente o desenho arquitetural descrito no `readme.md` e no `architecture.md`.