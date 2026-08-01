// build.rs (Salvo na raiz da sua crate que contém o executor)
fn main() {
    // 1. Informa ao cargo para buscar a biblioteca do driver CUDA no caminho padrão do Linux/HPC
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    // 2. Vincula dinamicamente com a libcuda.so instalada no nó do supercomputador
    println!("cargo:rustc-link-lib=dylib=cuda");

    // Opcional: Se você for usar a ferramenta 'bindgen' para gerar os cabeçalhos .h do CUDA em tempo de compilação:
    // let bindings = bindgen::Builder::default().header("/usr/local/cuda/include/cuda.h").generate()...
}
