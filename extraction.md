Com base na estrutura do repositório `lazyparser/FlagTree` [0†L2-L4], os arquivos que você precisa extrair para o **Basalto** estão organizados nas seguintes categorias:

---

## 1. 🧠 Núcleo do Compilador (FLIR + Passes + Análises)

Extraia **todo o conteúdo** destes diretórios — eles formam a base do `basalto-core`:

| Diretório no FlagTree | O que contém | Destino no Basalto |
| :--- | :--- | :--- |
| [`include/triton/`](https://github.com/lazyparser/FlagTree/tree/main/include/triton) | Cabeçalhos C++ da IR, tipos, operações, análise e suporte a MLIR. | `crates/basalto-core/src/ir/`, `analysis/` |
| [`lib/Analysis/`](https://github.com/lazyparser/FlagTree/tree/main/lib/Analysis) | Passes de análise (aliasing, liveness, etc.). | `crates/basalto-core/src/analysis/` |
| [`lib/Conversion/`](https://github.com/lazyparser/FlagTree/tree/main/lib/Conversion) | Lowering de TTIR → FLIR → dialetos de hardware. | `crates/basalto-core/src/conversion/` |
| [`lib/Dialect/`](https://github.com/lazyparser/FlagTree/tree/main/lib/Dialect) | Definição dos dialetos MLIR (TTIR, FLIR). | `crates/basalto-core/src/dialect/` |
| [`lib/Target/`](https://github.com/lazyparser/FlagTree/tree/main/lib/Target) | Infraestrutura de codegen (base para os backends). | `crates/basalto-target-*/` (parcial) |
| [`lib/Tools/`](https://github.com/lazyparser/FlagTree/tree/main/lib/Tools) | Utilitários auxiliares (ex.: suporte a layouts lineares). | `crates/basalto-common/src/` (parcial) |

**Arquivos específicos a extrair (além dos diretórios acima):**
- `include/CMakeLists.txt` [5†L9-L10] — apenas para referência de dependências.
- `lib/CMakeLists.txt` [6†L27-L28] — para mapear bibliotecas linkadas.

---

## 2. 🎯 Backends (NVIDIA, AMD, Intel)

Extraia **apenas** os backends que você vai suportar. Cada um está em `third_party/<backend>/`:

| Backend | Diretório no FlagTree | O que extrair | Destino no Basalto |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | [`third_party/nvidia/`](https://github.com/lazyparser/FlagTree/tree/main/third_party/nvidia) | `backend/`, `include/`, `lib/`, `triton_nvidia.cc` | `crates/basalto-target-nvidia/` |
| **AMD** | [`third_party/amd/`](https://github.com/lazyparser/FlagTree/tree/main/third_party/amd) | Mesma estrutura (backend/, include/, lib/) | `crates/basalto-target-amd/` |
| **Intel** | Ainda não está no branch `main` como backend explícito — mas o suporte a SPIR-V/OneAPI pode ser extraído do `lib/Target/` genérico ou do branch `triton_v3.2.x` (se disponível). | Codegen para SPIR-V via LLVM. | `crates/basalto-target-intel/` |

**Arquivos a ignorar:** backends de outros fabricantes (`iluvatar/`, `mthreads/`, `metax/`, `hcu/`, `cambricon/`, etc.) — você não vai usá-los.

---

## 3. 🐍 Bindings Python e Integração com PyTorch

Extraia a camada que conecta o compilador ao ecossistema Python:

| Diretório no FlagTree | O que contém | Destino no Basalto |
| :--- | :--- | :--- |
| [`python/triton/`](https://github.com/lazyparser/FlagTree/tree/main/python/triton) | Código Python do Triton (AST, linguagem, ops, autotune). | `python/basalto/ops/` (parcial) e `basalto-tree` (interceptor) |
| [`python/src/`](https://github.com/lazyparser/FlagTree/tree/main/python/src) | Bindings C++ (pybind11) que expõem o compilador ao Python. | Será **substituído** por PyO3 em `crates/basalto-tree/` — mas a lógica de interface pode ser aproveitada. |
| [`python/setup_tools/`](https://github.com/lazyparser/FlagTree/tree/main/python/setup_tools) | Scripts de build e configuração do pacote Python. | `pyproject.toml` + `maturin` (não extrair diretamente). |

**Arquivos específicos a extrair:**
- `python/triton/compiler.py` — lógica de compilação JIT (base para o `interceptor.rs`).
- `python/triton/ops/` — kernels e autotune (úteis para testes e referência).
- `python/triton/language/` — a DSL que o usuário escreve (pode ser mantida ou substituída).

---

## 4. 📦 Dependências e Build (referência)

Estes arquivos **não são extraídos** para o código-fonte, mas servem como referência para recriar o ambiente de build:

| Arquivo | Uso |
| :--- | :--- |
| [`python/requirements.txt`](https://github.com/lazyparser/FlagTree/blob/main/python/requirements.txt) | Dependências Python (ex.: torch, pytest). |
| [`CMakeLists.txt` (raiz)](https://github.com/lazyparser/FlagTree/blob/main/CMakeLists.txt) | Estrutura de build — será substituída por `Cargo.toml`. |
| [`documents/build.md`](https://github.com/lazyparser/FlagTree/blob/main/documents/build.md) | Instruções de compilação para cada backend — útil para mapear flags e dependências de SDK. |

---

## 5. 🗑️ O que NÃO extrair

- `third_party/iluvatar/`, `mthreads/`, `metax/`, `hcu/`, `cambricon/`, etc. — backends não utilizados.
- `python/examples/`, `tutorials/` — apenas documentação.
- `docs/`, `documents/` — podem ser consultados, mas não fazem parte do código-fonte.
- `reports/`, `utils/` — não são núcleo do compilador.

---

## ✅ Resumo da Extração

| Categoria | Diretórios/Arquivos a extrair | Destino |
| :--- | :--- | :--- |
| **Núcleo (C++)** | `include/triton/`, `lib/Analysis/`, `lib/Conversion/`, `lib/Dialect/`, `lib/Target/`, `lib/Tools/` | `crates/basalto-core/` e `basalto-common/` |
| **Backend NVIDIA** | `third_party/nvidia/` (backend/, include/, lib/) | `crates/basalto-target-nvidia/` |
| **Backend AMD** | `third_party/amd/` (backend/, include/, lib/) | `crates/basalto-target-amd/` |
| **Backend Intel** | `lib/Target/` (genérico) + branch específico (se houver) | `crates/basalto-target-intel/` |
| **Python/Triton** | `python/triton/` (compiler.py, language/, ops/) | `python/basalto/` (adaptado) |
| **Bindings C++** | `python/src/` (apenas como referência) | Será reescrito em PyO3 em `crates/basalto-tree/` |

---

Agora você tem o mapa exato do que pegar do FlagTree. Se quiser, posso detalhar **arquivo por arquivo** de um desses diretórios (ex.: listar todos os `.cpp` e `.h` dentro de `lib/Analysis/`) para facilitar a extração. Basta apontar qual parte você quer começar.