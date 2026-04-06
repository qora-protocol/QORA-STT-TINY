//! Whisper encoder: Conv1D stem + transformer layers.
//!
//! Input: mel spectrogram [80, 3000]
//! Output: encoder hidden states [1500, 384]

use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};
use crate::weights::{EncoderWeights, EncoderLayerWeights};

/// Cached AVX-512 detection: 0 = unchecked, 1 = no, 2 = yes.
static AVX512_DETECTED: AtomicU8 = AtomicU8::new(0);

#[inline]
fn use_avx512() -> bool {
    match AVX512_DETECTED.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let has = crate::simd::has_avx512();
            AVX512_DETECTED.store(if has { 2 } else { 1 }, Ordering::Relaxed);
            if has {
                eprintln!("[SIMD] AVX-512F detected — using accelerated kernels");
            }
            has
        }
    }
}

/// Run the full encoder forward pass.
/// mel: [n_mels, n_frames] = [80, 3000] in channel-first format.
/// Returns [seq_len, d_model] = [1500, 384].
pub fn encoder_forward(
    mel: &[f32],
    weights: &EncoderWeights,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_mels = 80;
    let n_frames = 3000;

    // 1. Conv1D stem
    // Conv1: [80, 3000] → [384, 3000], kernel=3, stride=1, pad=1
    let after_conv1 = conv1d(mel, n_mels, n_frames, &weights.conv1_w, &weights.conv1_b, 3, 1, 1);
    let conv1_len = n_frames;
    let after_conv1 = gelu_vec(&after_conv1);

    // Conv2: [384, 3000] → [384, 1500], kernel=3, stride=2, pad=1
    let after_conv2 = conv1d(&after_conv1, d_model, conv1_len, &weights.conv2_w, &weights.conv2_b, 3, 2, 1);
    let seq_len = (conv1_len + 2 * 1 - 3) / 2 + 1; // = 1500
    let after_conv2 = gelu_vec(&after_conv2);

    // Transpose from [d_model, seq_len] to [seq_len, d_model] + add positional embeddings
    let mut hidden = vec![0.0f32; seq_len * d_model];
    for t in 0..seq_len {
        for d in 0..d_model {
            hidden[t * d_model + d] = after_conv2[d * seq_len + t]
                + weights.embed_positions[t * d_model + d];
        }
    }

    // 3. Encoder layers
    for (i, layer) in weights.layers.iter().enumerate() {
        eprintln!("  Encoder layer {}/{}", i + 1, weights.layers.len());
        hidden = encoder_layer_forward(&hidden, layer, seq_len, d_model, n_heads, head_dim);
    }

    // 4. Final layer norm
    for t in 0..seq_len {
        let offset = t * d_model;
        layer_norm_inplace(&mut hidden[offset..offset + d_model], &weights.ln_w, &weights.ln_b);
    }

    hidden
}

/// Single encoder layer: LN → self-attn → residual → LN → FFN → residual.
fn encoder_layer_forward(
    input: &[f32],
    layer: &EncoderLayerWeights,
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = input.to_vec();

    // Self-attention block
    {
        let mut normed = input.to_vec();
        for t in 0..seq_len {
            let off = t * d_model;
            layer_norm_inplace(&mut normed[off..off + d_model], &layer.sa_ln_w, &layer.sa_ln_b);
        }

        let attn_out = self_attention(
            &normed, seq_len, d_model, n_heads, head_dim,
            &layer.q_proj_w, &layer.q_proj_b,
            &layer.k_proj_w, // k_proj has NO bias in Whisper
            &layer.v_proj_w, &layer.v_proj_b,
            &layer.o_proj_w, &layer.o_proj_b,
        );

        for i in 0..output.len() {
            output[i] += attn_out[i];
        }
    }

    // FFN block
    {
        let mut normed = output.clone();
        for t in 0..seq_len {
            let off = t * d_model;
            layer_norm_inplace(&mut normed[off..off + d_model], &layer.ff_ln_w, &layer.ff_ln_b);
        }

        let ffn_out = ffn_forward(&normed, seq_len, d_model, &layer.fc1_w, &layer.fc1_b, &layer.fc2_w, &layer.fc2_b);

        for i in 0..output.len() {
            output[i] += ffn_out[i];
        }
    }

    output
}

// ============================================================
// Core operations
// ============================================================

/// Conv1D: input [in_ch, in_len] → output [out_ch, out_len].
/// Weight shape: [out_ch, in_ch, kernel_size].
fn conv1d(
    input: &[f32], in_ch: usize, in_len: usize,
    weight: &[f32], bias: &[f32],
    kernel_size: usize, stride: usize, padding: usize,
) -> Vec<f32> {
    let out_ch = bias.len();
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    // Parallelize over output channels
    let output: Vec<f32> = (0..out_ch).into_par_iter().flat_map(|oc| {
        let mut row = vec![0.0f32; out_len];
        for t in 0..out_len {
            let mut sum = bias[oc];
            let in_start = t * stride;
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let in_pos = in_start + k;
                    let val = if in_pos >= padding && in_pos < in_len + padding {
                        input[ic * in_len + (in_pos - padding)]
                    } else {
                        0.0
                    };
                    sum += val * weight[oc * in_ch * kernel_size + ic * kernel_size + k];
                }
            }
            row[t] = sum;
        }
        row
    }).collect();

    output
}

/// Multi-head self-attention (encoder, non-causal).
/// input: [seq_len, d_model], weights already transposed to [d_model, d_model] (row-major).
/// Note: k_proj has NO bias in Whisper.
pub fn self_attention(
    input: &[f32], seq_len: usize, d_model: usize,
    n_heads: usize, head_dim: usize,
    q_w: &[f32], q_b: &[f32],
    k_w: &[f32], // NO bias
    v_w: &[f32], v_b: &[f32],
    o_w: &[f32], o_b: &[f32],
) -> Vec<f32> {
    // Project Q, K, V
    let q = gemm_bias(input, seq_len, d_model, q_w, d_model, q_b);
    let k = gemm_nobias(input, seq_len, d_model, k_w, d_model);
    let v = gemm_bias(input, seq_len, d_model, v_w, d_model, v_b);

    // Multi-head attention — parallelize over heads
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_outputs: Vec<Vec<f32>> = (0..n_heads).into_par_iter().map(|h| {
        let h_off = h * head_dim;
        let mut head_out = vec![0.0f32; seq_len * head_dim];

        for qi in 0..seq_len {
            let q_base = qi * d_model + h_off;
            let mut scores = vec![0.0f32; seq_len];
            for ki in 0..seq_len {
                let k_base = ki * d_model + h_off;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + d] * k[k_base + d];
                }
                scores[ki] = dot * scale;
            }

            softmax_inplace(&mut scores);

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..seq_len {
                    sum += scores[ki] * v[ki * d_model + h_off + d];
                }
                head_out[qi * head_dim + d] = sum;
            }
        }

        head_out
    }).collect();

    // Reassemble heads into [seq_len, d_model]
    let mut attn_output = vec![0.0f32; seq_len * d_model];
    for (h, head_out) in head_outputs.iter().enumerate() {
        let h_off = h * head_dim;
        for qi in 0..seq_len {
            for d in 0..head_dim {
                attn_output[qi * d_model + h_off + d] = head_out[qi * head_dim + d];
            }
        }
    }

    // Output projection
    gemm_bias(&attn_output, seq_len, d_model, o_w, d_model, o_b)
}

/// FFN: Linear(d→4d) → GELU → Linear(4d→d).
fn ffn_forward(
    input: &[f32], seq_len: usize, d_model: usize,
    fc1_w: &[f32], fc1_b: &[f32],
    fc2_w: &[f32], fc2_b: &[f32],
) -> Vec<f32> {
    let ffn_dim = fc1_b.len();

    // FC1: [seq, d_model] → [seq, ffn_dim]
    let mut h = gemm_bias(input, seq_len, d_model, fc1_w, ffn_dim, fc1_b);

    // GELU activation
    for v in &mut h {
        *v = gelu(*v);
    }

    // FC2: [seq, ffn_dim] → [seq, d_model]
    gemm_bias(&h, seq_len, ffn_dim, fc2_w, d_model, fc2_b)
}

// ============================================================
// Math primitives
// ============================================================

/// GEMM with bias: A[m, k] @ W[k, n] + bias[n] → out[m, n].
/// W is stored row-major as [k, n] (pre-transposed from [n, k]).
/// Parallelized over rows with cache-friendly i-p-j loop order.
pub fn gemm_bias(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, bias: &[f32]) -> Vec<f32> {
    let avx512 = use_avx512();
    let mut out = vec![0.0f32; m * n];

    // Parallelize over rows
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        // Initialize with bias
        row.copy_from_slice(&bias[..n]);

        let a_row = &a[i * k..(i + 1) * k];

        #[cfg(target_arch = "x86_64")]
        if avx512 {
            unsafe { crate::simd::gemm_row_avx512(row, a_row, w, k, n) };
            return;
        }

        // Scalar fallback: i-p-j order
        for p in 0..k {
            let a_val = a_row[p];
            let w_row = &w[p * n..(p + 1) * n];
            for j in 0..n {
                row[j] += a_val * w_row[j];
            }
        }
    });

    out
}

/// GEMM without bias: A[m, k] @ W[k, n] → out[m, n].
pub fn gemm_nobias(a: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let avx512 = use_avx512();
    let mut out = vec![0.0f32; m * n];

    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k..(i + 1) * k];

        #[cfg(target_arch = "x86_64")]
        if avx512 {
            unsafe { crate::simd::gemm_row_avx512(row, a_row, w, k, n) };
            return;
        }

        // Scalar fallback
        for p in 0..k {
            let a_val = a_row[p];
            let w_row = &w[p * n..(p + 1) * n];
            for j in 0..n {
                row[j] += a_val * w_row[j];
            }
        }
    });

    out
}

/// GEMV without bias: x[k] @ W[k, n] → out[n].
pub fn gemv_nobias(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if use_avx512() {
        return unsafe { crate::simd::gemv_f32_avx512(x, w, k, n) };
    }

    let mut out = vec![0.0f32; n];
    for p in 0..k {
        let x_val = x[p];
        let w_row = &w[p * n..(p + 1) * n];
        for j in 0..n {
            out[j] += x_val * w_row[j];
        }
    }
    out
}

/// GEMV with bias: x[k] @ W[k, n] + bias[n] → out[n].
pub fn gemv_bias(x: &[f32], w: &[f32], k: usize, n: usize, bias: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if use_avx512() {
        return unsafe { crate::simd::gemv_f32_avx512_bias(x, w, k, n, bias) };
    }

    let mut out = bias.to_vec();
    for p in 0..k {
        let x_val = x[p];
        let w_row = &w[p * n..(p + 1) * n];
        for j in 0..n {
            out[j] += x_val * w_row[j];
        }
    }
    out
}

/// Layer normalization: x = (x - mean) / sqrt(var + eps) * gamma + beta.
pub fn layer_norm_inplace(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5f32).sqrt();

    for i in 0..n {
        x[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/// GELU activation.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| gelu(x)).collect()
}

pub fn softmax_inplace(x: &mut [f32]) {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}
