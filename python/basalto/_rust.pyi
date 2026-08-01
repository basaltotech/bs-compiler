import torch
import torch.fx
import basalto._rust as rust
import os

def basalto_backend(gm: torch.fx.GraphModule, example_inputs):
    if not example_inputs:
        return gm

    op = "stencil"
    fused_op = None

    nodes = list(gm.graph.nodes)
    for i, node in enumerate(nodes):
        if node.op == "call_function":
            target_str = str(node.target)
            if "matmul" in target_str or "mm" in target_str or "bmm" in target_str:
                op = "matmul"
                if i + 1 < len(nodes):
                    next_node = nodes[i + 1]
                    if next_node.op == "call_function":
                        next_target = str(next_node.target)
                        if "bias" in next_target:
                            fused_op = "bias"
                            if i + 2 < len(nodes) and "relu" in str(nodes[i + 2].target):
                                fused_op = "bias_relu"
                            elif i + 2 < len(nodes) and "gelu" in str(nodes[i + 2].target):
                                fused_op = "bias_gelu"
                        elif "relu" in next_target:
                            fused_op = "relu"
                        elif "gelu" in next_target:
                            fused_op = "gelu"
                        elif "scale" in next_target:
                            fused_op = "scale"
                break
            elif "attention" in target_str:
                op = "attention"
                break
            elif "softmax" in target_str:
                op = "softmax"
                break

    if op == "matmul":
        if len(example_inputs) < 2:
            return gm
        a = example_inputs[0]
        b = example_inputs[1]
        batch = a.shape[0] if a.dim() >= 3 else 1
        if a.dim() >= 3:
            m = a.shape[1]
            k = a.shape[2]
            n = b.shape[2] if b.dim() >= 3 else b.shape[1]
        else:
            m = a.shape[0]
            k = a.shape[1]
            n = b.shape[1]
        c = torch.empty((batch, m, n), device=a.device, dtype=a.dtype)

        if fused_op:
            op = f"matmul_{fused_op}"

        shape = [batch, m, k, n]
        dtype = str(a.dtype).split('.')[-1]
        strides = [0, 0, 0, 0]
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or os.environ.get("LSB_JOBID")

        interceptor = rust.PyBasaltoInterceptor()  # pyright: ignore[reportAttributeAccessIssue]
        interceptor.compile_and_execute(
            op=op,
            dtype=dtype,
            shape=shape,
            strides=strides,
            job_id=job_id,
            device_ptr_x=a.data_ptr(),
            device_ptr_y=c.data_ptr(),
            device_ptr_z=b.data_ptr(),
        )
        return gm

    x = example_inputs[0]
    y = torch.zeros_like(x)
    shape = list(x.shape)
    dtype = str(x.dtype).split('.')[-1]
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or os.environ.get("LSB_JOBID")
    strides = list(x.stride())

    interceptor = rust.PyBasaltoInterceptor()  # pyright: ignore[reportAttributeAccessIssue]
    interceptor.compile_and_execute(
        op=op,
        dtype=dtype,
        shape=shape,
        strides=strides,
        job_id=job_id,
        device_ptr_x=x.data_ptr(),
        device_ptr_y=y.data_ptr(),
        device_ptr_z=0,
    )
    return gm

torch._dynamo.register_backend("basalto", basalto_backend)  # pyright: ignore[reportArgumentType]