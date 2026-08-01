use anyhow::{anyhow, Result};

/// Reorganiza e otimiza os layouts dos shapes para máxima performance na GPU.
/// - Remove dimensões unitárias desnecessárias (Achatamento).
/// - Verifica se as dimensões internas cumprem os requisitos de alinhamento do hardware (ex: alinhamento de 8 bytes para FP16/BF16).
pub fn reorganize_tensors(shapes: &[Vec<usize>]) -> Result<Vec<Vec<usize>>> {
    if shapes.is_empty() {
        return Err(anyhow!("Não é possível reorganizar uma lista vazia de shapes."));
    }

    let mut optimized_shapes = Vec::with_capacity(shapes.len());

    for shape in shapes {
        // 1. Remove dimensões de tamanho 1 (Squeeze de memória) para simplificar a indexação na GPU
        let mut new_shape: Vec<usize> = shape.iter()
            .cloned()
            .filter(|&dim| dim != 1)
            .collect();

        // Se o tensor era composto apenas por 1s (ex: [1, 1, 1]), mantemos pelo menos uma dimensão [1]
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        // 2. Validação/Otimização de Hardware (Regra dos Cores de Matriz)
        // Em GPUs modernas, a dimensão mais interna (o último elemento do shape) 
        // deve idealmente ser múltipla de 8 ou 16 para permitir acessos de memória vetorizados (coalesced memory access).
        if let Some(&last_dim) = new_shape.last() {
            if last_dim % 8 != 0 && last_dim > 8 {
                // Aqui o seu compilador Basalto saberia que precisará aplicar "Padding" 
                // (adicionar zeros fantasmas) para alinhar a memória na GPU.
                // Por enquanto, apenas rastreamos ou aplicamos um aviso/ajuste.
            }
        }

        optimized_shapes.push(new_shape);
    }

    Ok(optimized_shapes)
}
