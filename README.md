basalto/                                      # Raiz do projeto
│
├── Cargo.toml                                # Workspace Rust (todos os crates)
├── Cargo.lock
├── pyproject.toml                            # Configuração do maturin (bindings Python)
├── .env.example                              # Exemplo: REDIS_CACHE_URL, LOG_LEVEL
├── .gitignore
├── README.md
├── LICENSE
│
├── crates/                                   # ⭐ Toda a lógica em Rust (núcleo e diferenciais)
│   │
│   ├── basalto-common/                       # Utilitários compartilhados (extraído e adaptado)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── hardware.rs                   # Detecção dinâmica via APIs (CUDA/HIP/ZE) – extraído do FlagTree
│   │       ├── config.rs                     # Leitura de .env
│   │       ├── error.rs                      # Tipos de erro unificados
│   │       ├── permissions.rs                # ⭐ Verificação de CAP_SYS_NICE e device nodes (sem root)
│   │       └── telemetry.rs                  # Helpers para envio assíncrono de métricas (base para o COUN)
│   │
│   ├── basalto-core/                         # ⭐ Núcleo do compilador – EXTRAÍDO do FlagTree (FLIR + Hasher)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── ir/                           # AST, Tipos, Blocos (extraído de flagtree/include/triton/ir/)
│   │       ├── analysis/                     # Análises (extraído de flagtree/lib/Analysis/)
│   │       ├── transforms/                   # Otimizações (extraído de flagtree/lib/Transforms/)
│   │       ├── dialect/                      # Definição da FLIR (TTIR + extensões) – extraído
│   │       ├── flir_builder.rs               # Geração de FLIR a partir da AST Python (extraído e adaptado)
│   │       └── hasher.rs                     # ⭐ SHA-256 com (estrutura, tipos, shape, vendor, arch, driver_version)
│   │
│   ├── basalto-gems/                         # ⭐ ESCRITO DO ZERO – Stride View (reorganização de memória sem cópia)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── stride_view.rs                # Reinterpretação de layout de tensores (baseline do ganho de performance)
│   │
│   ├── basalto-target-nvidia/                # Codegen FLIR → PTX – EXTRAÍDO do FlagTree
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── codegen.rs                    # Gera PTX via LLVM (extraído de flagtree/lib/Target/NVIDIA/)
│   │       └── runtime.rs                    # Chamadas CUDA para validação
│   │
│   ├── basalto-target-amd/                   # Codegen FLIR → HSACO – EXTRAÍDO do FlagTree
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── codegen.rs                    # Gera HSACO (extraído de flagtree/lib/Target/AMD/)
│   │       └── runtime.rs
│   │
│   ├── basalto-target-intel/                 # Codegen FLIR → SPIR-V – EXTRAÍDO do FlagTree
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── codegen.rs                    # Gera SPIR-V (extraído de flagtree/lib/Target/Intel/)
│   │       └── runtime.rs
│   │
│   ├── siliconforge-jit/                     # ⭐ ESCRITO DO ZERO – Autocalibração contínua em tempo real
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── profiler.rs                   # Roda em tokio::spawn (task assíncrona) – recalibra blocos matemáticos em background
│   │
│   ├── basalto-tree/                         # ⭐ INTERCEPTADOR PRINCIPAL – Registrado no torch.compile()
│   │   ├── Cargo.toml                        # Depende de basalto-core, gems, targets, common
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── interceptor.rs                # ⭐ Fluxo: raw_call → gems::stride_view() → core::hash() → cache_lookup → (compile/reuse) → executor
│   │       ├── local_cache.rs                # Cache L1 (in-memory + disco local) – caminho mais rápido
│   │       ├── cluster_cache.rs              # Cache L2 (Redis – somente binário). Consulta/escrita oportunista, não bloqueante
│   │       └── executor.rs                   # Dispara kernel na GPU local + NOTIFICA siliconforge-jit (assíncrono)
│   │
│   ├── energy-telemetry/                     # ⭐ ESCRITO DO ZERO – Medição de kWh e correlação (COUN)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── reader.rs                     # Leitura de sensores (IPMI, Redfish, API proprietária)
│   │       └── correlator.rs                 # ⭐ Amarra (hash, timestamp_início, timestamp_fim, kWh_delta) por execução
│   │
│   └── basalto-installer/                    # ⭐ ESCRITO DO ZERO – Binário Rust para instalação segura
│       ├── Cargo.toml                        # Reutiliza basalto-common (hardware + permissions)
│       └── src/
│           └── main.rs                       # Detecta hardware, negocia permissões mínimas, gera configuração
│
├── python/                                   # Bindings PyO3 – EXPÕE O BASALTO-TREE para o PyTorch
│   ├── Cargo.toml                            # Depende de basalto-tree
│   ├── pyproject.toml
│   ├── basalto/
│   │   ├── __init__.py                       # Importa o interceptor do Rust
│   │   ├── _rust.pyi                         # Stubs para tipagem
│   │   ├── ops/                              # Autotune e heurísticas (mantido em Python, mas substituível pelo SiliconForge)
│   │   │   ├── __init__.py
│   │   │   ├── matmul.py
│   │   │   └── attention.py
│   │   └── compiler.py                       # Wrapper que registra o backend no torch.compile()
│   └── tests/
│       ├── unit/
│       └── integration/                      # Testes com GPU real (local)
│
├── deploy/                                   # Implantação – APENAS instalador e configuração opcional do Redis
│   ├── installer/
│   │   └── install.sh                        # ⭐ Bootstrap fino (bash): baixa o binário Rust, checa SO, executa o basalto-installer
│   └── redis/                                # (Opcional) Configuração do cache compartilhado
│       └── redis.conf
│
├── scripts/                                  # Utilitários para desenvolvimento
│   ├── dev-setup.sh
│   └── run-tests.sh
│
├── .github/                                  # CI/CD
│   └── workflows/
│       ├── ci.yml                            # Lint, testes unitários (sem GPU), build wheels
│       └── integration.yml                   # Testes com GPU real (self-hosted)
│
└── docs/
    ├── architecture.md                       # Este desenho final
    ├── integration.md                        # Como registrar no PyTorch
    ├── cache_protocol.md                     # Especificação da chave do Redis (hash + arch + driver)
    └── energy_telemetry.md                   # Como medir kWh e faturar em COUN