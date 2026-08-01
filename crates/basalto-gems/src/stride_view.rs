use std::ops::Range;

/// Reinterpreta um tensor 1D como 2D sem copiar, alterando stride.
/// Retorna um novo par (ptr, offset, shape, strides) para o mesmo buffer.
pub fn reinterpret_strides_2d(
    ptr: *mut f64,
    len: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
) -> Option<(*mut f64, usize, Vec<usize>, Vec<usize>)> {
    if rows * cols != len { return None; }
    // Aplica o stride: o step entre linhas é row_stride, entre colunas é 1 (ou outro)
    // Exemplo: para transposição, row_stride = 1, col_stride = rows
    // Para simplificação, só retornamos os valores.
    Some((ptr, 0, vec![rows, cols], vec![row_stride, 1]))
}