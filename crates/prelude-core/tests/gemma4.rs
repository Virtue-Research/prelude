//! Correctness tests for the Gemma4 model implementation.

mod common;

use prelude_core::tensor::{DType, Device, Module, Result, Tensor};
use prelude_core::ops::traits::{VarlenParams, MaskType};

// ── Config parsing ──────────────────────────────────────────────────────

#[test]
fn gemma4_config_parse_minimal() {
    let json = serde_json::json!({
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
    });
    let cfg: prelude_core::models::gemma4::Gemma4Config =
        serde_json::from_value(json).expect("should parse minimal config");
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.intermediate_size, 512);
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.vocab_size, 262144); // default
    assert!(!cfg.enable_moe_block);
    assert!(!cfg.attention_k_eq_v);
    assert_eq!(cfg.num_kv_shared_layers, 0);
}

#[test]
fn gemma4_config_parse_full() {
    let json = serde_json::json!({
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 26,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 262144,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention",
        ],
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0
            }
        },
        "sliding_window": 512,
        "enable_moe_block": true,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 4096,
        "num_kv_shared_layers": 2,
        "attention_k_eq_v": true,
        "num_global_key_value_heads": 4,
        "hidden_size_per_layer_input": 128,
        "use_double_wide_mlp": true,
    });
    let cfg: prelude_core::models::gemma4::Gemma4Config =
        serde_json::from_value(json).expect("should parse full config");
    assert_eq!(cfg.num_hidden_layers, 26);
    assert!(cfg.enable_moe_block);
    assert_eq!(cfg.num_experts.unwrap(), 4);
    assert_eq!(cfg.top_k_experts.unwrap(), 2);
    assert_eq!(cfg.num_kv_shared_layers, 2);
    assert!(cfg.attention_k_eq_v);
    assert_eq!(cfg.num_global_key_value_heads.unwrap(), 4);
    assert_eq!(cfg.hidden_size_per_layer_input.unwrap(), 128);
    assert!(cfg.use_double_wide_mlp);
    // rope_parameters
    let rp = cfg.rope_parameters.as_ref().unwrap();
    assert!((rp["full_attention"].rope_theta - 1_000_000.0).abs() < 1.0);
    assert!((rp["sliding_attention"].rope_theta - 10_000.0).abs() < 1.0);
}

#[test]
fn gemma4_config_nested_text_config() {
    let json = serde_json::json!({
        "text_config": {
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        }
    });
    // The text_config extraction is handled by the registry meta, not deserialization itself.
    // Let's test the inner config directly.
    let text_cfg = json.get("text_config").unwrap();
    let cfg: prelude_core::models::gemma4::Gemma4Config =
        serde_json::from_value(text_cfg.clone()).expect("should parse text_config");
    assert_eq!(cfg.hidden_size, 256);
}

// ── Proportional RoPE ───────────────────────────────────────────────────

#[test]
fn gemma4_rope_vs_pytorch() -> Result<()> {
    // Test that our proportional RoPE matches HF's Gemma4RotaryEmbedding.
    let head_dim = 64;
    let seq_len = 8;
    let num_heads = 2;
    let partial_rotary_factor = 0.5; // 50% rotation, 50% identity
    let rope_theta = 10000.0;

    let x_data = common::pseudo_random(seq_len * num_heads * head_dim, 42.0);

    let ref_flat = require_pytorch_ref!(
        &[("x", &x_data)],
        &format!(r#"
import torch
import math

head_dim = {head_dim}
seq_len = {seq_len}
num_heads = {num_heads}
partial_rotary_factor = {partial_rotary_factor}
rope_theta = {rope_theta}

rotary_dim = int(head_dim * partial_rotary_factor)
rotary_dim = (rotary_dim // 2) * 2
rope_angles = rotary_dim // 2
nope_angles = (head_dim // 2) - rope_angles

# Gemma4 proportional RoPE: denominator is head_dim
freq_exponents = torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim
inv_freq = 1.0 / (rope_theta ** freq_exponents)
if nope_angles > 0:
    inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles)])

# Compute cos/sin
positions = torch.arange(seq_len, dtype=torch.float)
freqs = torch.outer(positions, inv_freq)
cos = freqs.cos()
sin = freqs.sin()

# Apply RoPE to x
x = read_input("x").reshape(1, seq_len, num_heads, head_dim)
x1 = x[..., :head_dim//2]
x2 = x[..., head_dim//2:]
# Standard neox-style rotation
cos_exp = cos.unsqueeze(0).unsqueeze(2)
sin_exp = sin.unsqueeze(0).unsqueeze(2)
out1 = x1 * cos_exp - x2 * sin_exp
out2 = x2 * cos_exp + x1 * sin_exp
out = torch.cat([out1, out2], dim=-1)
write_output(out.float())
"#)
    );

    // Our implementation
    let x = Tensor::from_vec(x_data.clone(), (seq_len, num_heads, head_dim), &Device::Cpu)?;
    let positions = Tensor::arange(0u32, seq_len as u32, &Device::Cpu)?;

    let cos_sin = make_gemma4_cos_sin(head_dim, seq_len, rope_theta, partial_rotary_factor, &Device::Cpu)?;
    let cos = &cos_sin.0;
    let sin = &cos_sin.1;

    // Apply rope_thd
    let x4 = x.reshape((1, seq_len, num_heads, head_dim))?;
    let cos_sel = cos.index_select(&positions, 0)?;
    let sin_sel = sin.index_select(&positions, 0)?;
    let out = x4.rope_thd(&cos_sel, &sin_sel)?;
    let out = out.reshape((seq_len * num_heads * head_dim,))?;

    let ours: Vec<f32> = out.to_vec1()?;
    common::assert_close(&ours, &ref_flat, 1e-4, "gemma4_rope_proportional");
    Ok(())
}

/// Helper to build cos/sin tables matching Gemma4RotaryEmbedding.
fn make_gemma4_cos_sin(
    head_dim: usize,
    max_seq_len: usize,
    rope_theta: f64,
    partial_rotary_factor: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let rotary_dim = ((head_dim as f64 * partial_rotary_factor) as usize / 2) * 2;
    let rope_angles = rotary_dim / 2;
    let nope_angles = (head_dim / 2).saturating_sub(rope_angles);

    let mut inv_freq: Vec<f32> = (0..rope_angles)
        .map(|i| 1f32 / rope_theta.powf((2 * i) as f64 / head_dim as f64) as f32)
        .collect();
    for _ in 0..nope_angles {
        inv_freq.push(0.0);
    }
    let half_dim = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), dev)?.to_dtype(DType::F32)?;
    let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?;
    let freqs = t.broadcast_mul(&inv_freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

// ── MLP (GeluAndMul) ────────────────────────────────────────────────────

#[test]
fn gemma4_mlp_gelu_and_mul_vs_pytorch() -> Result<()> {
    let hidden = 32;
    let intermediate = 64;
    let seq = 4;

    let x_data = common::pseudo_random(seq * hidden, 1.0);
    let gate_w_data = common::pseudo_random(intermediate * hidden, 2.0);
    let up_w_data = common::pseudo_random(intermediate * hidden, 3.0);
    let down_w_data = common::pseudo_random(hidden * intermediate, 4.0);

    let ref_flat = require_pytorch_ref!(
        &[("x", &x_data), ("gate_w", &gate_w_data), ("up_w", &up_w_data), ("down_w", &down_w_data)],
        &format!(r#"
x = read_input("x").reshape({seq}, {hidden})
gate_w = read_input("gate_w").reshape({intermediate}, {hidden})
up_w = read_input("up_w").reshape({intermediate}, {hidden})
down_w = read_input("down_w").reshape({hidden}, {intermediate})

# Gemma4 MLP: GELU(gate) * up, then down
gate = x @ gate_w.T
up = x @ up_w.T
activated = torch.nn.functional.gelu(gate, approximate='tanh')
out = (activated * up) @ down_w.T
write_output(out.float())
"#)
    );

    let dev = Device::Cpu;
    let gate_w = Tensor::from_vec(gate_w_data, (intermediate, hidden), &dev)?;
    let up_w = Tensor::from_vec(up_w_data, (intermediate, hidden), &dev)?;
    let down_w = Tensor::from_vec(down_w_data, (hidden, intermediate), &dev)?;
    let x = Tensor::from_vec(x_data, (seq, hidden), &dev)?;

    let gate_proj = prelude_core::models::commons::linear::Linear::from_weight(gate_w, None)?;
    let up_proj = prelude_core::models::commons::linear::Linear::from_weight(up_w, None)?;
    let down_proj = prelude_core::models::commons::linear::Linear::from_weight(down_w, None)?;

    let ops = prelude_core::ops::select_ops(&dev);
    let bs = prelude_core::models::commons::BatchState::no_lora();

    let gate = gate_proj.forward(&x, &bs, ops)?;
    let up = up_proj.forward(&x, &bs, ops)?;
    let activated = gate.gelu()?;
    let out = down_proj.forward(&(activated * up)?, &bs, ops)?;

    let ours: Vec<f32> = out.reshape((seq * hidden,))?.to_vec1()?;
    // Large intermediate values (~183K) have absolute error ~0.05, relative is ~3e-7
    common::assert_close(&ours, &ref_flat, 1e-1, "gemma4_mlp");
    Ok(())
}

// ── Attention (scaling=1.0, Q/K/V norms) ────────────────────────────────

#[test]
fn gemma4_attention_qkv_norm_vs_pytorch() -> Result<()> {
    // Test that Q/K/V normalization + scaling=1.0 matches PyTorch.
    let hidden = 64;
    let head_dim = 32;
    let num_heads = 2;
    let seq = 4;

    let x_data = common::pseudo_random(seq * hidden, 10.0);
    let q_norm_w = common::pseudo_random(head_dim, 11.0);
    let k_norm_w = common::pseudo_random(head_dim, 12.0);

    let ref_flat = require_pytorch_ref!(
        &[("x", &x_data), ("q_norm_w", &q_norm_w), ("k_norm_w", &k_norm_w)],
        &format!(r#"
import torch
import math

x = read_input("x").reshape({seq}, {hidden})
q_norm_w = read_input("q_norm_w").reshape({head_dim})
k_norm_w = read_input("k_norm_w").reshape({head_dim})

# Simulate Q and K as raw projections (just use x reshaped)
q = x.reshape({seq}, {num_heads}, {head_dim})
k = x[:, :{head_dim}].reshape({seq}, 1, {head_dim}).expand({seq}, {num_heads}, {head_dim})

# RMSNorm per-head (Gemma4 Q/K norms)
def rms_norm(x, w, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (x * w.float()).to(x.dtype)

q_normed = rms_norm(q, q_norm_w)
k_normed = rms_norm(k, k_norm_w)

# Verify shapes
assert q_normed.shape == ({seq}, {num_heads}, {head_dim})
assert k_normed.shape == ({seq}, {num_heads}, {head_dim})

# Output Q and K normed flat
out = torch.cat([q_normed.reshape(-1), k_normed.reshape(-1)])
write_output(out.float())
"#)
    );

    let dev = Device::Cpu;
    let x = Tensor::from_vec(x_data, (seq, hidden), &dev)?;
    let q_norm_weight = Tensor::from_vec(q_norm_w, (head_dim,), &dev)?;
    let k_norm_weight = Tensor::from_vec(k_norm_w, (head_dim,), &dev)?;

    let q_norm = prelude_core::models::commons::linear::RmsNorm::from_weight(q_norm_weight, 1e-6);
    let k_norm = prelude_core::models::commons::linear::RmsNorm::from_weight(k_norm_weight, 1e-6);

    let q = x.reshape((seq, num_heads, head_dim))?;
    let k = x.narrow(1, 0, head_dim)?.reshape((seq, 1, head_dim))?
        .broadcast_as((seq, num_heads, head_dim))?.contiguous()?;

    let q_normed = q_norm.forward(&q)?;
    let k_normed = k_norm.forward(&k)?;

    let q_flat: Vec<f32> = q_normed.reshape((seq * num_heads * head_dim,))?.to_vec1()?;
    let k_flat: Vec<f32> = k_normed.reshape((seq * num_heads * head_dim,))?.to_vec1()?;
    let mut ours = q_flat;
    ours.extend(k_flat);

    common::assert_close(&ours, &ref_flat, 1e-4, "gemma4_qkv_norm");
    Ok(())
}

// ── V Norm (no learnable weight) ────────────────────────────────────────

#[test]
fn gemma4_v_norm_no_weight_vs_pytorch() -> Result<()> {
    let head_dim = 32;
    let seq = 4;
    let num_heads = 2;

    let v_data = common::pseudo_random(seq * num_heads * head_dim, 20.0);

    let ref_flat = require_pytorch_ref!(
        &[("v", &v_data)],
        &format!(r#"
import torch

v = read_input("v").reshape({seq}, {num_heads}, {head_dim})

# V norm: RMSNorm without learnable weight (weight=ones)
def rms_norm_no_weight(x, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)

v_normed = rms_norm_no_weight(v)
write_output(v_normed.float())
"#)
    );

    let dev = Device::Cpu;
    let v = Tensor::from_vec(v_data, (seq, num_heads, head_dim), &dev)?;
    let ones = Tensor::ones((head_dim,), DType::F32, &dev)?;
    let v_norm = prelude_core::models::commons::linear::RmsNorm::from_weight(ones, 1e-6);
    let v_normed = v_norm.forward(&v)?;
    let ours: Vec<f32> = v_normed.reshape((seq * num_heads * head_dim,))?.to_vec1()?;

    common::assert_close(&ours, &ref_flat, 1e-4, "gemma4_v_norm");
    Ok(())
}

// ── Full layer forward (small model, CPU) ───────────────────────────────

#[test]
fn gemma4_decoder_layer_vs_pytorch() -> Result<()> {
    // Small Gemma4 decoder layer: 2 heads, head_dim=16, hidden=32, intermediate=64
    let hidden = 32;
    let head_dim = 16;
    let num_heads = 2;
    let num_kv_heads = 1;
    let intermediate = 64;
    let seq = 3;

    // Generate deterministic weights
    let x_data = common::pseudo_random(seq * hidden, 100.0);
    let q_w = common::pseudo_random(num_heads * head_dim * hidden, 101.0);
    let k_w = common::pseudo_random(num_kv_heads * head_dim * hidden, 102.0);
    let v_w = common::pseudo_random(num_kv_heads * head_dim * hidden, 103.0);
    let o_w = common::pseudo_random(hidden * num_heads * head_dim, 104.0);
    let q_norm_w = common::pseudo_random(head_dim, 105.0);
    let k_norm_w = common::pseudo_random(head_dim, 106.0);
    let gate_w = common::pseudo_random(intermediate * hidden, 107.0);
    let up_w = common::pseudo_random(intermediate * hidden, 108.0);
    let down_w = common::pseudo_random(hidden * intermediate, 109.0);
    let ln1_w = common::pseudo_random(hidden, 110.0);
    let ln2_w = common::pseudo_random(hidden, 111.0);
    let ln3_w = common::pseudo_random(hidden, 112.0);
    let ln4_w = common::pseudo_random(hidden, 113.0);

    let ref_flat = require_pytorch_ref!(
        &[
            ("x", &x_data),
            ("q_w", &q_w), ("k_w", &k_w), ("v_w", &v_w), ("o_w", &o_w),
            ("q_norm_w", &q_norm_w), ("k_norm_w", &k_norm_w),
            ("gate_w", &gate_w), ("up_w", &up_w), ("down_w", &down_w),
            ("ln1_w", &ln1_w), ("ln2_w", &ln2_w), ("ln3_w", &ln3_w), ("ln4_w", &ln4_w),
        ],
        &format!(r#"
import torch
import math
import torch.nn.functional as F

hidden = {hidden}
head_dim = {head_dim}
num_heads = {num_heads}
num_kv_heads = {num_kv_heads}
intermediate = {intermediate}
seq = {seq}

x = read_input("x").reshape(seq, hidden)
q_w = read_input("q_w").reshape(num_heads * head_dim, hidden)
k_w = read_input("k_w").reshape(num_kv_heads * head_dim, hidden)
v_w = read_input("v_w").reshape(num_kv_heads * head_dim, hidden)
o_w = read_input("o_w").reshape(hidden, num_heads * head_dim)
q_norm_w = read_input("q_norm_w").reshape(head_dim)
k_norm_w = read_input("k_norm_w").reshape(head_dim)
gate_w = read_input("gate_w").reshape(intermediate, hidden)
up_w = read_input("up_w").reshape(intermediate, hidden)
down_w = read_input("down_w").reshape(hidden, intermediate)
ln1_w = read_input("ln1_w").reshape(hidden)
ln2_w = read_input("ln2_w").reshape(hidden)
ln3_w = read_input("ln3_w").reshape(hidden)
ln4_w = read_input("ln4_w").reshape(hidden)

def rms_norm(x, w, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (x * w.float()).to(x.dtype)

def rms_norm_no_weight(x, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)

# Build RoPE (standard, full rotation, theta=10000)
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
positions = torch.arange(seq, dtype=torch.float)
freqs = torch.outer(positions, inv_freq)
cos = freqs.cos()
sin = freqs.sin()

def apply_rope(q, k):
    # q: [seq, heads, head_dim], k: [seq, kv_heads, head_dim]
    def _rope(x):
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)
    return _rope(q), _rope(k)

# Layer forward: input_norm → attn → post_attn_norm + residual → pre_ff_norm → mlp → post_ff_norm + residual
residual = x.clone()

# 1. Input LayerNorm
h = rms_norm(residual, ln1_w)

# 2. Attention
q = (h @ q_w.T).reshape(seq, num_heads, head_dim)
k = (h @ k_w.T).reshape(seq, num_kv_heads, head_dim)
v = (h @ v_w.T).reshape(seq, num_kv_heads, head_dim)

q = rms_norm(q, q_norm_w)
k = rms_norm(k, k_norm_w)
v_ones = torch.ones(head_dim)
v = rms_norm_no_weight(v)

q, k = apply_rope(q, k)

# Expand KV for GQA
num_groups = num_heads // num_kv_heads
k = k.unsqueeze(2).expand(seq, num_kv_heads, num_groups, head_dim).reshape(seq, num_heads, head_dim)
v = v.unsqueeze(2).expand(seq, num_kv_heads, num_groups, head_dim).reshape(seq, num_heads, head_dim)

# Attention with scaling=1.0, causal mask
q_t = q.transpose(0, 1)  # [heads, seq, head_dim]
k_t = k.transpose(0, 1)
v_t = v.transpose(0, 1)
scores = torch.matmul(q_t, k_t.transpose(-2, -1))  # scaling=1.0
mask = torch.triu(torch.full((seq, seq), float('-inf')), diagonal=1)
scores = scores + mask.unsqueeze(0)
attn_weights = F.softmax(scores, dim=-1)
attn_out = torch.matmul(attn_weights, v_t)  # [heads, seq, head_dim]
attn_out = attn_out.transpose(0, 1).reshape(seq, num_heads * head_dim)

o = attn_out @ o_w.T

# 3. Post-attention norm + residual
o = rms_norm(o, ln2_w)
h = o + residual
residual = h.clone()

# 4. MLP
h = rms_norm(h, ln3_w)
gate = h @ gate_w.T
up = h @ up_w.T
activated = F.gelu(gate, approximate='tanh')
mlp_out = (activated * up) @ down_w.T

# 5. Post-FF norm + residual
mlp_out = rms_norm(mlp_out, ln4_w)
out = mlp_out + residual

# Layer scalar = 1.0 (default)
write_output(out.float())
"#)
    );

    // Build our Gemma4 layer using raw tensors
    let dev = Device::Cpu;
    let ops = prelude_core::ops::select_ops(&dev);
    let bs = prelude_core::models::commons::BatchState::no_lora();

    let x = Tensor::from_vec(x_data, (seq, hidden), &dev)?;

    // Create projections
    let q_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(q_w, (num_heads * head_dim, hidden), &dev)?, None)?;
    let k_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(k_w, (num_kv_heads * head_dim, hidden), &dev)?, None)?;
    let v_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(v_w, (num_kv_heads * head_dim, hidden), &dev)?, None)?;
    let o_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(o_w, (hidden, num_heads * head_dim), &dev)?, None)?;

    let q_norm = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::from_vec(q_norm_w, (head_dim,), &dev)?, 1e-6);
    let k_norm = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::from_vec(k_norm_w, (head_dim,), &dev)?, 1e-6);
    let v_norm = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::ones((head_dim,), DType::F32, &dev)?, 1e-6);

    let gate_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(gate_w, (intermediate, hidden), &dev)?, None)?;
    let up_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(up_w, (intermediate, hidden), &dev)?, None)?;
    let down_proj = prelude_core::models::commons::linear::Linear::from_weight(
        Tensor::from_vec(down_w, (hidden, intermediate), &dev)?, None)?;

    let ln1 = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::from_vec(ln1_w, (hidden,), &dev)?, 1e-6);
    let ln2 = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::from_vec(ln2_w, (hidden,), &dev)?, 1e-6);
    let ln3 = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::from_vec(ln3_w, (hidden,), &dev)?, 1e-6);
    let ln4 = prelude_core::models::commons::linear::RmsNorm::from_weight(
        Tensor::from_vec(ln4_w, (hidden,), &dev)?, 1e-6);

    // Build RoPE
    let (cos, sin) = make_gemma4_cos_sin(head_dim, seq, 10000.0, 1.0, &dev)?;
    let positions = Tensor::arange(0u32, seq as u32, &dev)?;
    let cos_sel = cos.index_select(&positions, 0)?;
    let sin_sel = sin.index_select(&positions, 0)?;

    // Forward: reproduce the layer
    let residual = x.clone();

    // 1. Input norm
    let h = ln1.forward(&residual)?;

    // 2. Attention projections
    let q = q_proj.forward(&h, &bs, ops)?;
    let k = k_proj.forward(&h, &bs, ops)?;
    let v = v_proj.forward(&h, &bs, ops)?;

    let q = q.reshape((seq, num_heads, head_dim))?;
    let k = k.reshape((seq, num_kv_heads, head_dim))?;
    let v = v.reshape((seq, num_kv_heads, head_dim))?;

    let q = q_norm.forward(&q)?;
    let k = k_norm.forward(&k)?;
    let v = v_norm.forward(&v)?;

    // Apply RoPE
    let q4 = q.reshape((1, seq, num_heads, head_dim))?;
    let k4 = k.reshape((1, seq, num_kv_heads, head_dim))?;
    let q = q4.rope_thd(&cos_sel, &sin_sel)?.reshape((seq, num_heads, head_dim))?;
    let k = k4.rope_thd(&cos_sel, &sin_sel)?.reshape((seq, num_kv_heads, head_dim))?;

    // varlen attention with scaling=1.0
    let cu_seqlens = Tensor::from_vec(vec![0u32, seq as u32], (2,), &dev)?;
    let attn_out = ops.varlen_attention(&q, &k, &v, &VarlenParams {
        cu_seqlens_q: &cu_seqlens,
        cu_seqlens_k: &cu_seqlens,
        max_seqlen_q: seq,
        max_seqlen_k: seq,
        scale: 1.0,
        mask: MaskType::Causal,
        softcap: None,
    })?;

    let o = o_proj.forward(&attn_out.reshape((seq, num_heads * head_dim))?, &bs, ops)?;

    // 3. Post-attention norm + residual
    let o = ln2.forward(&o)?;
    let h = (o + &residual)?;
    let residual = h.clone();

    // 4. MLP
    let h_normed = ln3.forward(&h)?;
    let gate = gate_proj.forward(&h_normed, &bs, ops)?;
    let up = up_proj.forward(&h_normed, &bs, ops)?;
    let activated = gate.gelu()?;
    let mlp_out = down_proj.forward(&(activated * up)?, &bs, ops)?;

    // 5. Post-FF norm + residual
    let mlp_out = ln4.forward(&mlp_out)?;
    let out = (mlp_out + &residual)?;

    let ours: Vec<f32> = out.reshape((seq * hidden,))?.to_vec1()?;
    common::assert_close(&ours, &ref_flat, 1e-3, "gemma4_decoder_layer");
    Ok(())
}

// ── Registry (tested via internal unit tests in meta module) ────────────

// ── Model forward (small config, random weights via VarBuilder::zeros) ──

#[test]
fn gemma4_model_forward_smoke() -> Result<()> {
    use prelude_core::loading::var_builder::VarBuilder;

    let cfg = prelude_core::models::gemma4::Gemma4Config {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 16,
        vocab_size: 100,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_local_base_freq: 10000.0,
        tie_word_embeddings: true,
        sliding_window: None,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        attention_bias: false,
        layer_types: Some(vec!["full_attention".into(), "sliding_attention".into()]),
        rope_parameters: None,
        global_head_dim: None,
        enable_moe_block: false,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        expert_intermediate_size: None,
        num_kv_shared_layers: 0,
        attention_k_eq_v: false,
        num_global_key_value_heads: None,
        hidden_size_per_layer_input: None,
        vocab_size_per_layer_input: None,
        use_double_wide_mlp: false,
    };

    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &dev);

    // Build model — zeros weights won't give meaningful output but tests shape/dispatch
    let mut model = prelude_core::models::gemma4::Gemma4ForCausalLM::new(&cfg, vb)?;

    let ops = prelude_core::ops::select_ops(&dev);
    let input_ids = Tensor::from_vec(vec![1u32, 5, 10], (3,), &dev)?;
    let position_ids = Tensor::from_vec(vec![0u32, 1, 2], (3,), &dev)?;
    let cu_seqlens = Tensor::from_vec(vec![0u32, 3], (2,), &dev)?;
    let seq_lens: Vec<usize> = vec![3];

    let mut ctx = prelude_core::models::commons::BatchAttnContext {
        ops,
        cu_seqlens_q: &cu_seqlens,
        max_seqlen_q: 3,
        position_ids: &position_ids,
        seq_lens: &seq_lens,
        paged_kv: None,
        deltanet_pool: None,
        deltanet_slots: None,
    };

    let logits = model.forward(&input_ids, &mut ctx)?;
    // Should output [1, 1, vocab_size] (last token selected, unsqueezed)
    let shape = logits.dims().to_vec();
    assert_eq!(shape.len(), 3, "logits should be 3D: {:?}", shape);
    assert_eq!(shape[0], 1, "batch=1");
    assert_eq!(shape[2], cfg.vocab_size, "last dim = vocab_size");

    model.clear_kv_cache();
    Ok(())
}

// ── Model with multiple sequences ──────────────────────────────────────

#[test]
fn gemma4_model_multi_seq_smoke() -> Result<()> {
    use prelude_core::loading::var_builder::VarBuilder;

    let cfg = prelude_core::models::gemma4::Gemma4Config {
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 16,
        vocab_size: 100,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_local_base_freq: 10000.0,
        tie_word_embeddings: true,
        sliding_window: Some(4),
        final_logit_softcapping: Some(30.0),
        attn_logit_softcapping: None,
        attention_bias: false,
        layer_types: Some(vec!["sliding_attention".into()]),
        rope_parameters: None,
        global_head_dim: None,
        enable_moe_block: false,
        num_experts: None,
        top_k_experts: None,
        moe_intermediate_size: None,
        expert_intermediate_size: None,
        num_kv_shared_layers: 0,
        attention_k_eq_v: false,
        num_global_key_value_heads: None,
        hidden_size_per_layer_input: None,
        vocab_size_per_layer_input: None,
        use_double_wide_mlp: false,
    };

    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &dev);

    let mut model = prelude_core::models::gemma4::Gemma4ForCausalLM::new(&cfg, vb)?;

    let ops = prelude_core::ops::select_ops(&dev);
    // Two sequences: [3 tokens] + [2 tokens] = 5 packed tokens
    let input_ids = Tensor::from_vec(vec![1u32, 5, 10, 20, 30], (5,), &dev)?;
    let position_ids = Tensor::from_vec(vec![0u32, 1, 2, 0, 1], (5,), &dev)?;
    let cu_seqlens = Tensor::from_vec(vec![0u32, 3, 5], (3,), &dev)?;
    let seq_lens: Vec<usize> = vec![3, 2];

    let mut ctx = prelude_core::models::commons::BatchAttnContext {
        ops,
        cu_seqlens_q: &cu_seqlens,
        max_seqlen_q: 3,
        position_ids: &position_ids,
        seq_lens: &seq_lens,
        paged_kv: None,
        deltanet_pool: None,
        deltanet_slots: None,
    };

    let logits = model.forward(&input_ids, &mut ctx)?;
    let shape = logits.dims().to_vec();
    // 2 sequences → 2 last tokens selected
    assert_eq!(shape[0], 2, "batch=2 sequences");
    assert_eq!(shape[2], cfg.vocab_size);

    Ok(())
}
