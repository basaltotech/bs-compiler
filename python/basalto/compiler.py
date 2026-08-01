import torch
import basalto._rust as rust  # importa a extensão PyO3

def basalto_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Interceptor para torch.compile.
    Converte o grafo FX em metadados e chama o Rust.
    """
    # Simplificação: extrai a primeira operação matmul/atenção
    # Em produção, percorre os nós do grafo.
    op = "matmul"
    dtype = str(example_inputs[0].dtype).split(".")[-1]
    shape = list(example_inputs[0].shape)
    job_id = None
    # Tenta ler SLURM_JOB_ID
    import os
    job_id = os.environ.get("SLURM_JOB_ID")

    interceptor = rust.BasaltoInterceptor()
    binary = interceptor.compile_and_execute(op, dtype, shape, job_id)
    # TODO: carregar o binário na GPU via PyTorch/CUDA
    # Por enquanto, retorna o grafo original (placeholder)
    return gm

# Registra no torch.compile
torch._dynamo.register_backend("basalto", basalto_backend)  # type: ignore