use basalto_target_nvidia::NvidiaRuntime;

let rt = NvidiaRuntime::new()?;
let ptx_bytes: Vec<u8> = ...; // vindo do codegen
let params: Vec<*const c_void> = vec![
    &device_ptr_a as *const _ as *const c_void,
    &device_ptr_b as *const _ as *const c_void,
    &n as *const _ as *const c_void,
];
rt.launch(&ptx_bytes, "my_kernel", (1,1,1), (256,1,1), 0, &params)?;