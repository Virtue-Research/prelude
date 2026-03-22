//! F32 CPU softmax — numerically stable row-wise softmax.

/// In-place row-wise softmax over a flat F32 buffer.
/// `data` layout is `[rows, cols]` row-major.
pub(crate) fn softmax_f32_inplace(data: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(data.len(), rows * cols);
    for row in data.chunks_exact_mut(cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        softmax_f32_inplace(&mut data, 1, 4);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for i in 0..3 {
            assert!(data[i] < data[i + 1]);
        }
    }

    #[test]
    fn test_softmax_with_neginf() {
        let mut data = vec![1.0, f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY];
        softmax_f32_inplace(&mut data, 1, 4);
        assert!(data[1] == 0.0);
        assert!(data[3] == 0.0);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_multirow() {
        let mut data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        softmax_f32_inplace(&mut data, 2, 3);
        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!((sum2 - 1.0).abs() < 1e-6);
    }
}
