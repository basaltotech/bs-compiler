Aqui está o `README.md` completo para a pasta `report/`, com instruções claras e detalhadas para que qualquer pesquisador ou engenheiro consiga rodar os benchmarks do Basalto em uma GPU alugada e gerar um relatório comparativo.

---

## 📄 `report/README.md`

```markdown
# Basalto – Benchmark Report

Este diretório contém tudo o que você precisa para executar uma bateria de benchmarks do Basalto em uma GPU alugada (RunPod, Lambda Labs, Vast.ai, etc.) e gerar um relatório comparativo.

---

## 📋 Pré‑requisitos

- Uma instância com GPU NVIDIA (A100, H100, V100, etc.)
- Ubuntu 22.04 LTS (recomendado) ou 20.04 LTS
- Acesso à internet para baixar dependências
- **Opcional:** Chave SSH para acesso remoto (já vem com a maioria dos provedores)

> **Custo estimado:** O benchmark completo leva menos de 2 horas. Com preços atuais (US$ 1,5–3,0/hora para A100), o custo total fica entre **US$ 3 e US$ 6**.

---

## 🚀 Passo a passo

### 1. Alugue uma instância GPU

Escolha um provedor especializado em GPU:

| Provedor | Preço A100 (aprox.) | Observação |
|----------|---------------------|------------|
| [RunPod](https://runpod.io) | US$ 1,39–1,49/hora | Boot rápido, suporte a imagens CUDA pré‑instaladas |
| [Lambda Labs](https://lambdalabs.com) | US$ 2,06/hora | Instâncias dedicadas, boa reputação |
| [Vast.ai](https://vast.ai) | US$ 1,5–2,0/hora | Mercado aberto, preços variáveis |
| [Jarvis Labs](https://jarvislabs.ai) | US$ 2,0–3,0/hora | Fácil de usar, suporte a Jupyter |

**Recomendação de imagem:**  
- Ubuntu 22.04 LTS
- CUDA 12.x (12.1 ou 12.2) com drivers NVIDIA
- PyTorch pré‑instalado (opcional, mas agiliza)

### 2. Conecte‑se à instância

```bash
ssh -i sua_chave.pem root@<IP_DA_INSTANCIA>
```

### 3. Clone o repositório e acesse a pasta `report`

```bash
git clone https://github.com/basaltotech/bs-compiler.git
cd bs-compiler/report
```

### 4. Execute o script de setup

```bash
chmod +x setup.sh
./setup.sh
```

O script instalará automaticamente:
- Rust, Cargo e maturin
- Python 3.10+ e pacotes (torch, numpy, etc.)
- Dependências do sistema (build-essential, curl, git)
- O Basalto (via `pip install -e ..` ou a partir do código)

> **Tempo estimado:** 5–10 minutos, dependendo da velocidade da rede.

### 5. Configure variáveis de ambiente (opcional)

```bash
cp config/.env.example .env
# Edite .env se quiser alterar parâmetros (ex: BASALTO_AUDIT_ENABLED)
```

Se não quiser configurar nada, os valores padrão serão usados.

### 6. Execute os benchmarks

```bash
python3 run_benchmarks.py --output results/benchmark_results.json
```

Isso executará, nesta ordem:

- **Stencil 1D/2D/3D** – compara Basalto vs Inductor (backend padrão do PyTorch).
- **MatMul denso** – compara Basalto vs Inductor vs PyTorch eager.
- **MatMul + fusão** (Bias, ReLU, GELU, Scale) – Basalto vs Inductor.
- **Medição de energia** – verifica se o COUN está funcionando (logs).

> **Tempo estimado:** 30–60 minutos, dependendo da GPU e dos tamanhos testados.

### 7. Gere o relatório

```bash
python3 generate_report.py --input results/benchmark_results.json --output report.md
```

O relatório será gerado em Markdown com:
- Tabelas de desempenho
- Speedup (aceleração) do Basalto em relação ao Inductor e ao eager
- Gráficos de barras (em texto)
- Conclusões

### 8. Analise o relatório

```bash
cat report.md
```

Ou copie o arquivo para sua máquina local:

```bash
scp -i sua_chave.pem root@<IP_DA_INSTANCIA>:~/bs-compiler/report/report.md .
```

---

## 📊 O que o relatório mostra

| Seção | Descrição |
|-------|-----------|
| **Stencils** | Tempo médio (ms) para stencils 1D, 2D e 3D. Speedup do Basalto vs Inductor. |
| **MatMul** | Tempo médio (ms) para MatMul denso e fundido. Speedup vs Inductor e vs eager. |
| **Energia** | Consumo de kWh registrado pelo Basalto (se `BASALTO_AUDIT_ENABLED=true`). |
| **Sistema** | Modelo da GPU, versão do CUDA, versão do PyTorch, etc. |

---

## 📈 Exemplo de saída esperada

```text
## Stencils

### 1D_1024_torch.float32

| Backend | Tempo (ms) | Speedup vs Inductor |
|---------|------------|---------------------|
| Basalto | 0.12       | 1.25x               |
| Inductor| 0.15       | 1.00x               |

### 3D_64x64x64_torch.float32

| Backend | Tempo (ms) | Speedup vs Inductor |
|---------|------------|---------------------|
| Basalto | 2.34       | 1.42x               |
| Inductor| 3.32       | 1.00x               |
```

---

## 🧠 Interpretação dos resultados

- **Speedup > 1.0** → Basalto é mais rápido que o baseline.
- **Speedup entre 1.2 e 1.4** → ganho significativo (típico para stencils com tiling).
- **Speedup < 1.0** → Basalto é mais lento (pode indicar incompatibilidade de driver ou configuração).

Se todos os testes passarem e o speedup for consistente, o Basalto está pronto para ser considerado em um piloto real.

---

## 🛠️ Solução de problemas

| Problema | Provável causa | Solução |
|----------|----------------|---------|
| `maturin build` falha | Rust não instalado ou versão antiga | Reinstale o Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` |
| `torch.cuda.is_available()` retorna `False` | Driver NVIDIA não carregado | Verifique `nvidia-smi`. Instale os drivers: `sudo apt install nvidia-driver-535` |
| `import basalto` falha | Wheel não instalada corretamente | Reconstrua: `maturin build --release && pip install target/wheels/basalto-*.whl` |
| Benchmark demora muito | GPU compartilhada ou shape muito grande | Reduza os shapes no script `stencil_benchmark.py` |

---

## 📝 Notas finais

- O benchmark é **auto‑contido** e não requer acesso à internet após o setup inicial.
- Os resultados são salvos em `results/benchmark_results.json` para análise posterior.
- Para testar em **multi‑GPU**, edite o arquivo `run_benchmarks.py` e adicione o argumento `--multi-gpu` (em desenvolvimento).

---

**Última atualização:** 2025

**Dúvidas ou sugestões:** entre em contato com a equipe Basalto Tech.
```

---

## ✅ O que este README cobre

| Seção | Descrição |
|-------|-----------|
| **Pré‑requisitos** | O que é necessário antes de começar (GPU, SO, acesso) |
| **Passo a passo** | Instruções claras e numeradas para executar o benchmark |
| **Exemplo de saída** | O que esperar do relatório |
| **Interpretação** | Como entender os números (speedup, ganhos) |
| **Solução de problemas** | Erros comuns e como resolvê‑los |
| **Notas finais** | Informações adicionais sobre o benchmark |
