import os
import torch 
from torch._dynamo import register_backend  # type: ignore

# O módulo `_rust` é gerado pelo maturin (PyO3) e instalado no pacote basalto.
# Em HPC, ele estará no mesmo diretório que este arquivo ou no site-packages.
try:
    from ._rust import basalto_tree
except ImportError:
    # Fallback para desenvolvimento: tenta importar do caminho absoluto (não recomendado em produção)
    import sys
    sys.path.append(os.path.dirname(__file__))
    from _rust import basalto_tree

def compile_from_fx_graph(graph: torch.fx.GraphModule, example_inputs):
    """
    Função registrada como backend do torch.compile.
    Executa UMA VEZ para compilar o grafo (JIT).
    Retorna um callable que será chamado a cada iteração.
    """
    # 1. Serializa o grafo (código Python gerado pelo Dynamo)
    graph_str = str(graph.code)

    # 2. Extrai as dimensões estáticas (shapes) dos exemplos de entrada
    #    Atenção: em HPC, shapes podem variar (batch dinâmico) – o hash BLAKE3
    #    no Rust já lida com isso, mas aqui passamos o que temos.
    shapes = [list(t.shape) for t in example_inputs if hasattr(t, "shape")]

    # 3. Metadados do hardware – em HPC, obtidos via variáveis de ambiente
    #    ou detectados pelo instalador (root). O Rust validará e sobrescreverá.
    vendor = os.getenv("BASALTO_VENDOR", "nvidia")
    arch   = os.getenv("BASALTO_ARCH", "sm_90")
    driver = os.getenv("BASALTO_DRIVER_VERSION", "12.8")

    # 4. Pipeline completo no Rust:
    #    - Basalto Gems (Stride View)
    #    - Hash BLAKE3 (com vendor/arch/driver)
    #    - Cache L1 (local) + L2 (Redis) – se habilitado
    #    - Codegen (PTX/HSACO/SPIR-V)
    #    - Execução síncrona na GPU (já dispara o kernel)
    #    - Telemetria em background (thread separada)
    binary = basalto_tree.compile_from_fx_graph(
        graph_str, shapes, vendor, arch, driver
    )

    # 5. FUNÇÃO DE RUNTIME (chamada a cada execução do kernel)
    #    Ela NÃO recompila – apenas dispara o binário pré-compilado na GPU.
    def run(*args):
        # Em HPC, os tensores de entrada são passados como argumentos.
        # O ideal é que o Rust já tenha o binário carregado e pronto.
        # Se quiser, pode chamar uma função do Rust para executar com novos dados:
        # return basalto_tree.execute_binary(binary, args)
        #
        # Porém, para garantir que o PyTorch não quebre o grafo,
        # retornamos a execução do grafo original (já substituído pelo backend).
        # MAS: se o backend já executou na GPU, chamar graph(*args) faria
        # a CPU esperar um resultado que já está na GPU – isso é ineficiente.
        # Em uma implementação real, você substituiria o `run` por uma função
        # que chama o kernel via CUDA/HIP diretamente.
        # Como placeholder, retornamos o resultado do grafo original (CPU).
        return graph(*args)

    return run

# Registra o backend globalmente para ser usado com @torch.compile(backend="basalto")
register_backend("basalto", compile_from_fx_graph)  # type: ignore