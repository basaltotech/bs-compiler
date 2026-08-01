import torch
import torch.fx
import basalto._rust as rust
import os

def basalto_backend(gm: torch.fx.GraphModule, example_inputs):
    if not example_inputs or not isinstance(example_inputs[0], torch.Tensor):
        return gm

    x = example_inputs[0]
    y = torch.zeros_like(x)

    shape = list(x.shape)
    strides = list(x.stride())
    dtype = str(x.dtype).split('.')[-1]
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or os.environ.get("LSB_JOBID")

    op = "stencil"
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_str = str(node.target)
            if "matmul" in target_str:
                op = "matmul"
                break
            elif "attention" in target_str:
                op = "attention"
                break
            elif "softmax" in target_str:
                op = "softmax"
                break

    interceptor = rust.PyBasaltoInterceptor()  # pyright: ignore[reportAttributeAccessIssue]
    interceptor.compile_and_execute(
        op=op,
        dtype=dtype,
        shape=shape,
        strides=strides,
        job_id=job_id,
        device_ptr_x=x.data_ptr(),
        device_ptr_y=y.data_ptr(),
    )
    return gm

torch._dynamo.register_backend("basalto", basalto_backend)  # pyright: ignore[reportArgumentType]