Não, você **não precisa** integrar nenhum agente específico para usar o Cursor. Ele já vem com um agente nativo poderoso e você pode potencializar seu uso com algumas configurações simples, como as **Cursor Rules**.

Aqui está um guia prático para otimizar sua experiência com o Cursor no desenvolvimento do Basalto.

---

### 🧠 1. Entendendo o "Agente" do Cursor

O **Cursor Agent** é o coração da ferramenta. Ele não é apenas um autocomplete; é uma IA que pode executar tarefas complexas de forma autônoma, como:
*   **Entender todo o seu código** (codebase-awareness).
*   **Criar e editar múltiplos arquivos**.
*   **Executar comandos no terminal** para você.
*   **Planejar e executar tarefas** complexas.
*   Até **executar vários agentes em paralelo** para explorar diferentes soluções.

Você acessa essa funcionalidade principalmente através do **Composer** (atalho `Cmd+I`), que é a interface principal para tarefas agenticas.

> **Resumo:** Você não precisa instalar nada. O agente é nativo e já está pronto para ser usado.

---

### ⚙️ 2. Como Otimizar o Cursor para o Basalto (Sem Agentes Externos)

A melhor maneira de otimizar o Cursor para seu projeto específico é usar **Cursor Rules**. Pense nelas como um "manual de instruções" que você dá para a IA, dizendo como você quer que ela se comporte, quais padrões de código seguir e quais ferramentas usar.

#### a) Crie um arquivo `.cursorrules` na raiz do projeto
Este arquivo contém instruções para guiar o comportamento da IA. Para o Basalto, você pode definir regras como:

```text
# Basalto Project Rules

## Tech Stack
- Primary language: Rust
- Python for bindings and benchmarks (PyO3)
- CUDA, cuBLAS, and CUTLASS for GPU acceleration
- MPI and NCCL for distributed communication

## Rust Guidelines
- Follow Rust API Guidelines
- Use `anyhow` and `thiserror` for error handling
- Prefer `tokio` for async runtime
- Use `libloading` for dynamic library loading (CUDA, MPI, NCCL)

## Python Guidelines
- Use `maturin` for building Python bindings
- Follow PEP 8 for Python code
- Use `torch` for tensor operations in benchmarks

## Project Structure
- `crates/` contains all Rust crates
- `python/` contains Python bindings
- `report/` contains benchmark scripts

## Communication
- When proposing changes, always consider cross-crate dependencies
- For GPU code, ensure compatibility with NVIDIA A100/H100
- For distributed code, consider MPI and NCCL implications
```

> **Dica:** Mantenha as regras concisas (menos de 500 linhas) e priorize as mais importantes no topo. Você também pode usar o formato moderno `.mdc` dentro da pasta `.cursor/rules/` para regras mais específicas.

#### b) Use o `@codebase` para contextualizar suas perguntas
Ao fazer uma pergunta no chat ou no Composer, marque `@codebase`. Isso força o Cursor a indexar e considerar **todo o seu projeto** para responder, em vez de apenas o arquivo aberto.

#### c) Aproveite o "Agent Mode" para tarefas complexas
Quando você tiver uma tarefa grande, como "refatore o módulo de comunicação para usar async", use o **Agent Mode** no Composer. Ele pode planejar, executar e até rodar múltiplos sub-agentes em paralelo para diferentes partes da tarefa.

---

### 💡 3. Integrações Úteis (Opcionais)

Embora não sejam agentes, algumas integrações podem turbinar seu fluxo de trabalho:

*   **Linear Integration:** Conecte seu projeto Linear ao Cursor. Você pode delegar tarefas diretamente de um issue para um agente do Cursor.
*   **Cursor Agent CLI:** Permite usar o agente do Cursor diretamente do terminal, útil para automação ou se você preferir a linha de comando.

---

### ✅ Resumo e Recomendação

Para o projeto Basalto, a receita é simples:

1.  **Crie um arquivo `.cursorrules`** na raiz do projeto com as diretrizes mencionadas (ajuste conforme a necessidade do seu time).
2.  **Use `@codebase`** no Chat e no Composer para todas as perguntas que envolvam múltiplos arquivos ou a arquitetura geral.
3.  **Explore o Agent Mode** para tarefas complexas de refatoração ou geração de código multi-arquivo.
4.  **Mantenha as regras atualizadas** conforme o projeto evolui. Equipes que mantêm boas regras relatam menos bugs e mais consistência no código.

Com essas práticas, você estará usando o Cursor no seu potencial máximo para o Basalto, sem precisar de nenhuma configuração de agente externo.