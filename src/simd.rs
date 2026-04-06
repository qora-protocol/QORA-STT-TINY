//! AVX-512 SIMD kernels for F32 GEMV/GEMM acceleration.
//!
//! Provides ~4x speedup over scalar code by processing 16 f32 values per cycle
//! using 512-bit FMA (fused multiply-add) instructions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Runtime detection of AVX-512F support.
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX-512 GEMV without bias: x[k] @ W[k, n] -> out[n].
/// W is stored row-major as [k, n].
///
/// # Safety
/// Caller must ensure AVX-512F is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_f32_avx512(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let n16 = n / 16 * 16;

    for p in 0..k {
        let x_val = x[p];
        if x_val == 0.0 {
            continue;
        }
        let val_vec = _mm512_set1_ps(x_val);
        let row = p * n;
        let mut j = 0usize;
        while j < n16 {
            let w_vec = _mm512_loadu_ps(w.as_ptr().add(row + j));
            let acc = _mm512_loadu_ps(out.as_ptr().add(j));
            _mm512_storeu_ps(out.as_mut_ptr().add(j), _mm512_fmadd_ps(val_vec, w_vec, acc));
            j += 16;
        }
        // Scalar tail
        while j < n {
            out[j] += x_val * w[row + j];
            j += 1;
        }
    }
    out
}

/// AVX-512 GEMV with bias: x[k] @ W[k, n] + bias[n] -> out[n].
///
/// # Safety
/// Caller must ensure AVX-512F is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_f32_avx512_bias(
    x: &[f32],
    w: &[f32],
    k: usize,
    n: usize,
    bias: &[f32],
) -> Vec<f32> {
    let mut out = bias.to_vec();
    let n16 = n / 16 * 16;

    for p in 0..k {
        let x_val = x[p];
        if x_val == 0.0 {
            continue;
        }
        let val_vec = _mm512_set1_ps(x_val);
        let row = p * n;
        let mut j = 0usize;
        while j < n16 {
            let w_vec = _mm512_loadu_ps(w.as_ptr().add(row + j));
            let acc = _mm512_loadu_ps(out.as_ptr().add(j));
            _mm512_storeu_ps(out.as_mut_ptr().add(j), _mm512_fmadd_ps(val_vec, w_vec, acc));
            j += 16;
        }
        // Scalar tail
        while j < n {
            out[j] += x_val * w[row + j];
            j += 1;
        }
    }
    out
}

/// AVX-512 inner loop for a single GEMM row: row[n] += a_val * w_row[n].
/// Called from within the rayon parallel iterator for each output row.
///
/// # Safety
/// Caller must ensure AVX-512F is available on the current CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemm_row_avx512(
    row: &mut [f32],
    a_row: &[f32],
    w: &[f32],
    k: usize,
    n: usize,
) {
    let n16 = n / 16 * 16;

    for p in 0..k {
        let a_val = a_row[p];
        if a_val == 0.0 {
            continue;
        }
        let val_vec = _mm512_set1_ps(a_val);
        let w_off = p * n;
        let mut j = 0usize;
        while j < n16 {
            let w_vec = _mm512_loadu_ps(w.as_ptr().add(w_off + j));
            let acc = _mm512_loadu_ps(row.as_ptr().add(j));
            _mm512_storeu_ps(row.as_mut_ptr().add(j), _mm512_fmadd_ps(val_vec, w_vec, acc));
            j += 16;
        }
        // Scalar tail
        while j < n {
            row[j] += a_val * w[w_off + j];
            j += 1;
        }
    }
}
