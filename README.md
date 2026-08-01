Para um **pesquisador** (geofísico, cientista de dados, engenheiro de simulação), o foco não é instalar ou configurar o sistema, mas **usar o Basalto para acelerar seus códigos** sem se preocupar com os detalhes internos. O guia abaixo é direto, prático e evita termos técnicos desnecessários.

---

## 📘 Guia do Pesquisador – Basalto Enterprise Suite

Parabéns! O Basalto já está instalado no cluster. Agora você pode acelerar suas simulações sísmicas e modelos de IA com um único comando.

---

### 1. O que o Basalto faz por você?

- **Compila automaticamente** seus kernels (stencils, MatMul, atenção, etc.) para a GPU exata que você está usando.
- **Reutiliza** o código compilado (cache) – a primeira execução é mais lenta, as seguintes são muito rápidas.
- **Otimiza o acesso à memória** (Stride View) sem você precisar mudar seu código.
- **Calibra em tempo real** (SiliconForge) ajustando parâmetros para melhor performance.
- **Mede o consumo de energia** (COUN) para que você possa auditá-lo, se desejar.

---

### 2. Como usar no PyTorch

É simples: basta importar o Basalto e usar o `torch.compile` com `backend="basalto"`.

```python
import torch
import basalto  # Isso já registra o backend "basalto"

# Sua função de simulação (exemplo: stencil 1D)
def meu_stencil(x):
    return (x[..., :-2] + x[..., 1:-1] + x[..., 2:]) / 3.0

# Compila a função para a GPU
funcao_rapida = torch.compile(meu_stencil, backend="basalto")

# Cria um tensor na GPU
dados = torch.randn(1000000, device="cuda")

# Executa – a primeira chamada compila, as seguintes são instantâneas
resultado = funcao_rapida(dados)
```

**Observação importante:** a primeira execução pode levar alguns segundos (é a compilação). Depois, tudo roda na velocidade máxima da GPU.

---

### 3. Exemplo prático – Multiplicação de matrizes (muito usada em IA)

```python
import torch
import basalto

def minha_rede(x, peso):
    return torch.matmul(x, peso)

x = torch.randn(512, 1024, device="cuda")
w = torch.randn(1024, 256, device="cuda")

modelo_rapido = torch.compile(minha_rede, backend="basalto")
y = modelo_rapido(x, w)  # usa cuBLAS (Tensor Cores se disponível)
```

---

### 4. Exemplo para simulação sísmica (stencil 3D com halos)

Se você trabalha com volumes sísmicos, o Basalto já entende stencils 3D com tiling e troca de halos.

```python
import torch
import basalto

def propagacao_onda(volume):
    # Stencil 3D simples (média dos vizinhos)
    # (aqui você coloca seu próprio stencil)
    return (volume[..., 1:-1, 1:-1, 1:-1] +
            volume[..., :-2, 1:-1, 1:-1] +
            volume[..., 2:, 1:-1, 1:-1]) / 3.0

volume = torch.randn(64, 64, 64, device="cuda")
propagacao = torch.compile(propagacao_onda, backend="basalto")
resultado = propagacao(volume)
```

O Basalto cuida automaticamente da **troca de halos** entre GPUs se você estiver rodando em múltiplos nós (MPI).

---

### 5. Dicas de performance

- **Primeira execução:** sempre será mais lenta (compilação). Use um **mini-batch** de teste para "aquecer" o cache antes do job real.
- **Cache persistente:** os kernels compilados ficam salvos em `/var/cache/basalto/kernels/`. Se você rodar o mesmo código novamente (mesmo dias depois), ele reutiliza o binário compilado – zero overhead.
- **Mude o tamanho dos dados:** se o tamanho dos tensores mudar, o Basalto recompila (pois a chave de cache inclui o shape). Isso é automático e transparente.

---

### 6. O que fazer se algo não funcionar

- **Verifique se o tensor está na GPU:** `x.device` deve ser `cuda`.
- **Confira o formato:** o Basalto funciona com `float32` e `float64`; para `float16`/`bfloat16`, ele ativa Tensor Cores automaticamente.
- **Logs:** se houver erro, o Basalto escreve mensagens no log do sistema (`/var/log/basalto/basalto.log`). Peça ajuda ao administrador para verificar.

---

### 7. Medição de energia (COUN)

Se a auditoria estiver habilitada, o Basalto registra o consumo de cada execução. Você não precisa fazer nada – o sistema já faz isso. Se quiser ver os dados:

```python
# Exemplo: acessar o correlator (avançado)
# (normalmente isso é usado pelo time de faturamento)
```

---

### 8. Suporte

- Para dúvidas sobre **uso científico** (como escrever stencils eficientes), procure a equipe de suporte técnico.
- Para **problemas de instalação ou performance**, entre em contato com a administração do cluster.

---

### Resumo para o pesquisador

| Ação | Comando / Procedimento |
|------|------------------------|
| Importar o Basalto | `import basalto` |
| Compilar uma função | `torch.compile(minha_funcao, backend="basalto")` |
| Rodar normalmente | A primeira execução compila; as demais são rápidas |
| Verificar logs | Peça ajuda ao admin (logs em `/var/log/basalto/`) |

Agora você pode aproveitar todo o poder do hardware sem se preocupar com compiladores, otimizações ou configurações. Basta escrever seu código científico como sempre fez – o Basalto faz o resto.