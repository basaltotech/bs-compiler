import torch
import torch.fx
import basalto._rust as rust
import os

def basalto_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Interceptor para torch.compile.
    Percorre o grafo FX e extrai a primeira operação relevante.
    """
    # Simplificação: pega o primeiro nó que não seja de entrada/saída
    op = "matmul"  # default
    dtype = "f32"
    shape = []

    for node in gm.graph.nodes:
        if node.op == "call_function":
            # Tentar identificar matmul, atenção, etc.
            if "matmul" in str(node.target):
                op = "matmul"
            elif "attention" in str(node.target):
                op = "attention"
            elif "softmax" in str(node.target):
                op = "softmax"
            # Pega o shape do primeiro argumento tensor
            if len(node.args) > 0 and isinstance(node.args[0], torch.fx.Node):
                # Não temos shape facilmente; pegamos de example_inputs
                pass
            break

    # Usa o primeiro tensor de entrada para dtype/shape
    if example_inputs and isinstance(example_inputs[0], torch.Tensor):
        dtype = str(example_inputs[0].dtype).split('.')[-1]
        shape = list(example_inputs[0].shape)
        # Ponteiros dos tensores (assumindo que estão na GPU)
        # Para um kernel simples 1D, pegamos o primeiro tensor como entrada e criamos um de saída
        x = example_inputs[0]
        y = torch.zeros_like(x)  # saída placeholder
        n = x.numel()
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or os.environ.get("LSB_JOBID")

        interceptor = rust.PyBasaltoInterceptor()  # pyright: ignore[reportAttributeAccessIssue]
        interceptor.compile_and_execute(
            op=op,
            dtype=dtype,
            shape=shape,
            job_id=job_id,
            device_ptr_x=x.data_ptr(),
            device_ptr_y=y.data_ptr(),
            n=n,
        )
        # Retorna o grafo original, pois a execução já foi feita.
        return gm
    else:
        return gm

# Registra o backend
torch._dynamo.register_backend("basalto", basalto_backend) # type: ignore