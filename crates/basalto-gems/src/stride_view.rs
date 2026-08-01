use std::fmt;
use std::ops::Index;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrideView<T> {
    data: *mut T,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    len: usize,
}

impl<T> StrideView<T> {
    pub fn new(data: *mut T, shape: Vec<usize>, strides: Vec<isize>) -> Option<Self> {
        if shape.len() != strides.len() {
            return None;
        }
        let len = shape.iter().product();
        Some(Self {
            data,
            shape,
            strides,
            offset: 0,
            len,
        })
    }

    pub fn from_slice(data: &mut [T], shape: Vec<usize>) -> Option<Self> {
        let strides: Vec<isize> = shape
            .iter()
            .rev()
            .scan(1, |acc, &dim| {
                let s = *acc;
                *acc *= dim as isize;
                Some(s)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        Self::new(data.as_mut_ptr(), shape, strides)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1;
        for (dim, stride) in self.shape.iter().zip(self.strides.iter()).rev() {
            if *stride != expected as isize {
                return false;
            }
            expected *= dim;
        }
        true
    }

    pub fn index_linear(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut idx = self.offset as isize;
        for (i, &dim) in indices.iter().enumerate() {
            if dim >= self.shape[i] {
                return None;
            }
            idx += dim as isize * self.strides[i];
        }
        if idx < 0 {
            return None;
        }
        Some(idx as usize)
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let idx = self.index_linear(indices)?;
        unsafe { self.data.add(idx).as_ref() }
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let idx = self.index_linear(indices)?;
        unsafe { self.data.add(idx).as_mut() }
    }

    pub fn transpose(&self) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let mut new_strides = self.strides.clone();
        new_strides.reverse();
        Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            len: self.len,
        }
    }

    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Option<Self> {
        if dim >= self.shape.len() {
            return None;
        }
        if start > end || end > self.shape[dim] {
            return None;
        }
        let mut new_shape = self.shape.clone();
        new_shape[dim] = end - start;
        let offset = self.offset + start * self.strides[dim] as usize;
        Some(Self {
            data: self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset,
            len: self.len,
        })
    }

    pub fn as_ptr(&self) -> *const T {
        unsafe { self.data.add(self.offset) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.data.add(self.offset) }
    }

    pub fn iter(&self) -> StrideViewIterator<T> {
        StrideViewIterator {
            view: self,
            indices: vec![0; self.shape.len()],
            done: false,
        }
    }

    pub fn optimize_for_gpu(&self, warp_size: usize) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        let mut new_strides = self.strides.clone();
        for (i, stride) in new_strides.iter_mut().enumerate() {
            if *stride < 0 {
                *stride = -stride;
            }
            if i == new_strides.len() - 1 {
                *stride = (*stride / warp_size as isize + 1) * warp_size as isize;
            }
        }
        Self {
            data: self.data,
            shape: self.shape.clone(),
            strides: new_strides,
            offset: self.offset,
            len: self.len,
        }
    }

    pub fn coalesce(&self) -> Option<Self> {
        if self.is_contiguous() {
            return Some(self.clone());
        }
        let mut dims: Vec<usize> = (0..self.shape.len()).collect();
        dims.sort_by_key(|&i| self.strides[i].abs());
        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_strides = Vec::with_capacity(self.shape.len());
        let mut current_stride = 1;
        for &i in dims.iter().rev() {
            new_shape.push(self.shape[i]);
            new_strides.push(current_stride);
            current_stride *= self.shape[i] as isize;
        }
        new_shape.reverse();
        new_strides.reverse();
        Some(Self {
            data: self.data,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            len: self.len,
        })
    }
}

impl<T> Clone for StrideView<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            len: self.len,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for StrideView<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StrideView {{ shape: {:?}, strides: {:?}, offset: {}, len: {} }}",
            self.shape, self.strides, self.offset, self.len)
    }
}

pub struct StrideViewIterator<'a, T> {
    view: &'a StrideView<T>,
    indices: Vec<usize>,
    done: bool,
}

impl<'a, T> Iterator for StrideViewIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = self.view.get(&self.indices);
        let mut i = self.indices.len();
        while i > 0 {
            i -= 1;
            self.indices[i] += 1;
            if self.indices[i] < self.view.shape[i] {
                return result;
            }
            self.indices[i] = 0;
        }
        self.done = true;
        result
    }
}

pub fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    if shape.is_empty() {
        return true;
    }
    let mut expected = 1;
    for (dim, stride) in shape.iter().zip(strides.iter()).rev() {
        if *stride != expected as isize {
            return false;
        }
        expected *= dim;
    }
    true
}

pub fn optimal_strides(shape: &[usize]) -> Vec<isize> {
    shape
        .iter()
        .rev()
        .scan(1, |acc, &dim| {
            let s = *acc;
            *acc *= dim as isize;
            Some(s)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

pub fn coalesce_strides(shape: &[usize], strides: &[isize]) -> Option<(Vec<usize>, Vec<isize>)> {
    if shape.len() != strides.len() {
        return None;
    }
    let mut dims: Vec<usize> = (0..shape.len()).collect();
    dims.sort_by_key(|&i| strides[i].abs());
    let mut new_shape = Vec::with_capacity(shape.len());
    let mut new_strides = Vec::with_capacity(shape.len());
    let mut current_stride = 1;
    for &i in dims.iter().rev() {
        new_shape.push(shape[i]);
        new_strides.push(current_stride);
        current_stride *= shape[i] as isize;
    }
    new_shape.reverse();
    new_strides.reverse();
    Some((new_shape, new_strides))
}