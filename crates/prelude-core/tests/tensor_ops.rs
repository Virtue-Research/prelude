//! Core tensor operation tests -- verifies correctness of our own tensor implementation.
//! Test values verified against reference implementations.
//!
//! Compute ops (matmul, softmax, activations, reductions, unary math) are tested
//! across F32/BF16/F16 with PyTorch reference. Shape/logic tests stay f32-only.

mod common;

use prelude_core::tensor::{DType, Device, Result, Tensor, TensorExt, D};
use prelude_core::ops::{self, traits::{Ops, VarlenParams, MaskType}};

/// Register device ops once, then return the test device.
fn dev() -> &'static Device {
    use std::sync::{LazyLock, Once};
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        prelude_cpu::register();
        #[cfg(feature = "test-cuda")]
        prelude_cuda::register();
    });
    static DEV: LazyLock<Device> = LazyLock::new(|| common::test_devices().last().unwrap().clone());
    &DEV
}

/// Dtype configs to test for matmul-based tests. Candle CPU matmul is F32-only,
/// so BF16/F16 are skipped unless the test device is CUDA.
fn matmul_dtypes() -> &'static [&'static common::DTypeConfig] {
    if dev().is_cuda() { common::ALL_DTYPES } else { &[&common::F32_CONFIG] }
}

/// Run a test body on ALL test devices (CPU + GPU if enabled).
macro_rules! for_each_device {
    ($body:expr) => {{
        for __dev in common::test_devices().iter() {
            (|| -> Result<()> {
                let dev: &Device = __dev;
                let _ = dev;
                $body
            })()
            .map_err(|e| prelude_core::tensor::Error::Msg(
                format!("[device {:?}] {e}", __dev).into()
            ))?;
        }
        Ok(())
    }};
}

// == Tensor creation ==

#[test]
fn zeros() -> Result<()> {
    let t = Tensor::zeros((5, 2), DType::F32, dev())?;
    assert_eq!(t.dims(), &[5, 2]);
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    assert!(v.iter().all(|&x| x == 0.0));
    Ok(())
}

#[test]
fn ones() -> Result<()> {
    let t = Tensor::ones((2, 3), DType::F32, dev())?;
    let v: Vec<Vec<f32>> = t.to_vec2()?;
    assert_eq!(v, vec![vec![1.0; 3]; 2]);
    Ok(())
}

// == Arithmetic ==

#[test]
fn add_mul() -> Result<()> {
    let t = Tensor::from_vec(vec![3.0f32, 1.0, 4.0], 3, dev())?;
    let sum = (&t + &t)?;
    assert_eq!(sum.to_vec1::<f32>()?, vec![6.0, 2.0, 8.0]);
    let prod = (&sum * &sum)?;
    assert_eq!(prod.to_vec1::<f32>()?, vec![36.0, 4.0, 64.0]);
    Ok(())
}

#[test]
fn binary_ops() -> Result<()> {
    let t1 = Tensor::from_vec(
        vec![3f32, 1., 4., 1., 5., 2., 1., 7., 8., 2.],
        (2, 5),
        dev(),
    )?;
    let t2 = Tensor::from_vec(
        vec![5f32, 5., 5., 5., 5., 2., 1., 7., 8., 2.],
        (2, 5),
        dev(),
    )?;
    // t1 + (t1 * t1) / (t1 + t2)
    let result = (&t1 + &(&t1 * &t1)?.broadcast_div(&(&t1 + &t2)?)?)?;
    let v = common::to_vec2_round(&result, 4)?;
    assert_eq!(
        v,
        vec![
            vec![4.125, 1.1667, 5.7778, 1.1667, 7.5],
            vec![3.0, 1.5, 10.5, 12.0, 3.0],
        ]
    );
    Ok(())
}

#[test]
fn minimum_maximum() -> Result<()> {
    let t1 = Tensor::from_vec(
        vec![3f32, 1., 4., 1., 5., 2., 1., 7., 8., 2.],
        (2, 5),
        dev(),
    )?;
    let t2 = Tensor::from_vec(
        vec![5f32, 5., 5., 5., 5., 2., 1., 7., 8., 2.],
        (2, 5),
        dev(),
    )?;
    let half_t2 = (&t2 * 0.5)?;
    let min_v = common::to_vec2_round(&t1.minimum(&half_t2)?, 1)?;
    assert_eq!(
        min_v,
        vec![vec![2.5, 1.0, 2.5, 1.0, 2.5], vec![1.0, 0.5, 3.5, 4.0, 1.0],]
    );
    let max_v = common::to_vec2_round(&t1.maximum(&half_t2)?, 1)?;
    assert_eq!(
        max_v,
        vec![vec![3.0, 2.5, 4.0, 2.5, 5.0], vec![2.0, 1.0, 7.0, 8.0, 2.0],]
    );
    Ok(())
}

// == Unary / Activation ops ==

#[test]
fn activations() -> Result<()> {
    let t = Tensor::from_vec(
        vec![-3f32, 1., 4., -0.1, 0.5, 2.7, -1.8, -0.28, 1.8, 2.8],
        (2, 5),
        dev(),
    )?;

    // SiLU
    let silu = t.silu()?;
    let v = common::to_vec2_round(&silu, 4)?;
    assert_eq!(
        v,
        vec![
            vec![-0.1423, 0.7311, 3.9281, -0.0475, 0.3112],
            vec![2.53, -0.2553, -0.1205, 1.5447, 2.6395],
        ]
    );

    // GELU (approximate)
    let gelu = t.gelu()?;
    let v = common::to_vec2_round(&gelu, 4)?;
    assert_eq!(
        v,
        vec![
            vec![-0.0036, 0.8412, 3.9999, -0.046, 0.3457],
            vec![2.6911, -0.0647, -0.1091, 1.7353, 2.7933],
        ]
    );

    // GELU (erf)
    let gelu_erf = t.gelu_erf()?;
    let v = common::to_vec2_round(&gelu_erf, 4)?;
    assert_eq!(
        v,
        vec![
            vec![-0.004, 0.8413, 3.9999, -0.046, 0.3457],
            vec![2.6906, -0.0647, -0.1091, 1.7353, 2.7928],
        ]
    );

    Ok(())
}

#[test]
fn elu() -> Result<()> {
    let t = Tensor::from_vec(vec![-1.0f32, 0.0, -2.0, 3.0], 4, dev())?;
    let result = t.elu(2.0)?;
    let v = common::to_vec1_round(&result, 4)?;
    assert_eq!(v, vec![-1.2642, 0.0, -1.7293, 3.0]);
    Ok(())
}

#[test]
fn clamp() -> Result<()> {
    let t = Tensor::from_vec(
        vec![3f32, 1., 4., 1., 5., 2., 1., 7., 8., 2.],
        (2, 5),
        dev(),
    )?;
    let c = t.clamp(1.5, 6.2)?;
    let v = common::to_vec2_round(&c, 1)?;
    assert_eq!(
        v,
        vec![vec![3.0, 1.5, 4.0, 1.5, 5.0], vec![2.0, 1.5, 6.2, 6.2, 2.0],]
    );
    Ok(())
}

// == Matmul ==

#[test]
fn matmul_2x2() -> Result<()> {
    let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), dev())?;
    let b = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), dev())?;
    let c = a.matmul(&b)?;
    assert_eq!(c.to_vec2::<f32>()?, vec![vec![7.0, 10.0], vec![15.0, 22.0]]);
    Ok(())
}

#[test]
fn matmul_2x3_3x2() -> Result<()> {
    let a = Tensor::from_vec(vec![0f32, 1., 2., 3., 4., 5.], (2, 3), dev())?;
    let b = Tensor::from_vec(vec![2f32, 3., 4., 5., 6., 7.], (3, 2), dev())?;
    let c = a.matmul(&b)?;
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![vec![16.0, 19.0], vec![52.0, 64.0]]
    );
    Ok(())
}

#[test]
fn matmul_batched() -> Result<()> {
    let a: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let b: Vec<f32> = (2..14).map(|x| x as f32).collect();
    let a = Tensor::from_vec(a, (2, 2, 3), dev())?;
    let b = Tensor::from_vec(b, (2, 3, 2), dev())?;
    let c = a.matmul(&b)?;
    assert_eq!(
        c.to_vec3::<f32>()?,
        vec![
            vec![vec![16.0, 19.0], vec![52.0, 64.0]],
            vec![vec![214.0, 235.0], vec![304.0, 334.0]],
        ]
    );
    Ok(())
}

#[test]
fn matmul_vs_pytorch() -> Result<()> {
    let m = 8;
    let k = 256;
    let n = 128;
    let a_data = common::pseudo_random(m * k, 1.0);
    let b_data = common::pseudo_random(k * n, 2.0);

    for cfg in matmul_dtypes() {
        let reference = require_pytorch_ref!(
            &[("a", &a_data), ("b", &b_data)],
            &format!(r#"
a = read_input("a", {py}).reshape({m}, {k})
b = read_input("b", {py}).reshape({k}, {n})
y = (a @ b).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let a = Tensor::from_vec(a_data.clone(), (m, k), dev())?.to_dtype(cfg.dtype)?;
        let b = Tensor::from_vec(b_data.clone(), (k, n), dev())?.to_dtype(cfg.dtype)?;
        let y = a.matmul(&b)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_matmul,
            &format!("matmul {:?}", cfg.dtype),
        );
    }
    Ok(())
}

#[test]
fn matmul_tall_skinny_vs_pytorch() -> Result<()> {
    // Decode-like: [1, 1024] @ [1024, 1024]
    let m = 1;
    let k = 1024;
    let n = 1024;
    let a_data = common::pseudo_random(m * k, 3.0);
    let b_data = common::pseudo_random(k * n, 4.0);

    for cfg in matmul_dtypes() {
        let reference = require_pytorch_ref!(
            &[("a", &a_data), ("b", &b_data)],
            &format!(r#"
a = read_input("a", {py}).reshape({m}, {k})
b = read_input("b", {py}).reshape({k}, {n})
y = (a @ b).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let a = Tensor::from_vec(a_data.clone(), (m, k), dev())?.to_dtype(cfg.dtype)?;
        let b = Tensor::from_vec(b_data.clone(), (k, n), dev())?.to_dtype(cfg.dtype)?;
        let y = a.matmul(&b)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_matmul,
            &format!("matmul_tall_skinny {:?}", cfg.dtype),
        );
    }
    Ok(())
}

/// Gemma4 double-wide MLP dimensions: (7, 1536) × (1536, 12288)
/// Tests the exact CUTLASS GEMM path that Gemma4's KV-shared layers use.
/// SM90 skips (n=7 < 64), SM80 bf16_c1 handles it.
#[test]
fn matmul_gemma4_double_wide_mlp_vs_pytorch() -> Result<()> {
    let m = 7;    // seq_len (prefill)
    let k = 1536; // hidden_size
    let n = 12288; // 2× intermediate_size
    let a_data = common::pseudo_random(m * k, 7.0);
    let b_data = common::pseudo_random(k * n, 8.0);

    for cfg in matmul_dtypes() {
        let reference = require_pytorch_ref!(
            &[("a", &a_data), ("b", &b_data)],
            &format!(r#"
a = read_input("a", {py}).reshape({m}, {k})
b = read_input("b", {py}).reshape({k}, {n})
y = (a @ b).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let a = Tensor::from_vec(a_data.clone(), (m, k), dev())?.to_dtype(cfg.dtype)?;
        let b = Tensor::from_vec(b_data.clone(), (k, n), dev())?.to_dtype(cfg.dtype)?;
        let y = a.matmul(&b)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_matmul,
            &format!("matmul_gemma4_double_wide {:?} [{m},{k}]×[{k},{n}]", cfg.dtype),
        );
    }
    Ok(())
}

/// Gemma4 attention: (7, 512) × (512, 7) — Q@K^T score computation
#[test]
fn matmul_gemma4_attn_score_vs_pytorch() -> Result<()> {
    if !dev().is_cuda() {
        eprintln!("SKIPPED: BF16 matmul requires CUDA");
        return Ok(());
    }
    let m = 7;
    let k = 512; // head_dim=512
    let n = 7;   // seq_len
    let a_data = common::pseudo_random(m * k, 9.0);
    let b_data = common::pseudo_random(k * n, 10.0);

    for cfg in [&common::BF16_CONFIG] {
        let reference = require_pytorch_ref!(
            &[("a", &a_data), ("b", &b_data)],
            &format!(r#"
a = read_input("a", {py}).reshape({m}, {k})
b = read_input("b", {py}).reshape({k}, {n})
y = (a @ b).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let a = Tensor::from_vec(a_data.clone(), (m, k), dev())?.to_dtype(cfg.dtype)?;
        let b = Tensor::from_vec(b_data.clone(), (k, n), dev())?.to_dtype(cfg.dtype)?;
        let y = a.matmul(&b)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_matmul,
            &format!("matmul_gemma4_attn_score {:?} [{m},{k}]×[{k},{n}]", cfg.dtype),
        );
    }
    Ok(())
}

/// Chained matmul precision: simulates 14 layers of linear projections
/// to verify error accumulation stays bounded. Uses Gemma4-like dimensions.
#[test]
fn matmul_chained_14_layers_vs_pytorch() -> Result<()> {
    if !dev().is_cuda() {
        eprintln!("SKIPPED: BF16 matmul requires CUDA");
        return Ok(());
    }
    let seq = 7;
    let hidden = 1536;
    let inter = 6144;
    // Simulate: x → gate_proj → gelu → down_proj, repeated 14 times
    let x_data = common::pseudo_random(seq * hidden, 11.0);
    let gate_data = common::pseudo_random(hidden * inter, 12.0);
    let down_data = common::pseudo_random(inter * hidden, 13.0);

    for cfg in [&common::BF16_CONFIG] {
        let reference = require_pytorch_ref!(
            &[("x", &x_data), ("gate", &gate_data), ("down", &down_data)],
            &format!(r#"
import torch
x = read_input("x", {py}).reshape({seq}, {hidden})
gate_w = read_input("gate", {py}).reshape({inter}, {hidden})
down_w = read_input("down", {py}).reshape({hidden}, {inter})
for i in range(14):
    h = x @ gate_w.t()
    h = torch.nn.functional.gelu(h, approximate='tanh')
    h = h @ down_w.t()
    x = x + h * 0.5  # residual with scaling
x = x.float()
write_output(x)
"#,
                py = cfg.py_dtype,
            )
        );

        let x = Tensor::from_vec(x_data.clone(), (seq, hidden), dev())?.to_dtype(cfg.dtype)?;
        let gate_w = Tensor::from_vec(gate_data.clone(), (inter, hidden), dev())?.to_dtype(cfg.dtype)?;
        let down_w = Tensor::from_vec(down_data.clone(), (hidden, inter), dev())?.to_dtype(cfg.dtype)?;

        let mut x_cur = x;
        for _ in 0..14 {
            let h = x_cur.matmul(&gate_w.t()?)?;
            let h = h.gelu()?;
            let h = h.matmul(&down_w.t()?)?;
            x_cur = (x_cur + (h * 0.5)?)?;
        }
        let ours: Vec<f32> = x_cur.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_chained,
            &format!("matmul_chained_14_layers {:?}", cfg.dtype),
        );
    }
    Ok(())
}

// == Reductions ==

#[test]
fn sum_keepdim() -> Result<()> {
    let t = Tensor::from_vec(
        vec![3u32, 1, 4, 1, 5, 9, 2, 1, 7, 8, 2, 8],
        (2, 2, 3),
        dev(),
    )?;
    // Sum along last dim
    let s = t.sum_keepdim(2)?;
    assert_eq!(
        s.to_vec3::<u32>()?,
        vec![vec![vec![8], vec![15]], vec![vec![10], vec![18]]]
    );
    // Sum along batch dim
    let s = t.sum_keepdim(0)?;
    assert_eq!(
        s.to_vec3::<u32>()?,
        vec![vec![vec![5, 2, 11], vec![9, 7, 17]]]
    );
    Ok(())
}

#[test]
fn max_min_keepdim() -> Result<()> {
    let t = Tensor::from_vec(
        vec![3u32, 1, 4, 1, 5, 9, 2, 1, 7, 8, 2, 8],
        (2, 2, 3),
        dev(),
    )?;
    let mx = t.max_keepdim(2)?;
    assert_eq!(
        mx.to_vec3::<u32>()?,
        vec![vec![vec![4], vec![9]], vec![vec![7], vec![8]]]
    );
    let mn = t.min_keepdim(2)?;
    assert_eq!(
        mn.to_vec3::<u32>()?,
        vec![vec![vec![1], vec![1]], vec![vec![1], vec![2]]]
    );
    Ok(())
}

#[test]
fn argmax_argmin() -> Result<()> {
    let t = Tensor::from_vec(
        vec![3u32, 1, 4, 1, 5, 9, 2, 1, 7, 8, 2, 8],
        (2, 2, 3),
        dev(),
    )?;
    let am = t.argmax(2)?;
    assert_eq!(am.to_vec2::<u32>()?, vec![vec![2, 2], vec![2, 0]]);
    let ami = t.argmin(2)?;
    assert_eq!(ami.to_vec2::<u32>()?, vec![vec![1, 0], vec![1, 1]]);
    Ok(())
}

#[test]
fn sum_large() -> Result<()> {
    let v: Vec<u32> = (0..4000).collect();
    let t = Tensor::from_vec(v, 4000, dev())?;
    let s = t.sum_all()?;
    assert_eq!(s.to_scalar::<u32>()?, 7998000);
    Ok(())
}

#[test]
fn sum_vs_pytorch() -> Result<()> {
    let n = 10000;
    let data = common::pseudo_random(n, 15.0);

    for cfg in common::ALL_DTYPES {
        let reference = require_pytorch_ref!(
            &[("x", &data)],
            &format!(r#"
x = read_input("x", {py})
write_output(x.float().sum().reshape(1))
"#,
                py = cfg.py_dtype
            )
        );

        let x = Tensor::from_vec(data.clone(), (n,), dev())?.to_dtype(cfg.dtype)?;
        let s = x.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            (s - reference[0]).abs() < cfg.atol_reduction * 100.0,
            "sum {:?}: ours={s}, pytorch={}, diff={}",
            cfg.dtype,
            reference[0],
            (s - reference[0]).abs()
        );
    }
    Ok(())
}

#[test]
fn mean_vs_pytorch() -> Result<()> {
    let rows = 64;
    let cols = 512;
    let data = common::pseudo_random(rows * cols, 16.0);

    for cfg in common::ALL_DTYPES {
        let reference = require_pytorch_ref!(
            &[("x", &data)],
            &format!(r#"
x = read_input("x", {py}).reshape({rows}, {cols})
y = x.float().mean(dim=1)
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let x = Tensor::from_vec(data.clone(), (rows, cols), dev())?.to_dtype(cfg.dtype)?;
        let y = x.to_dtype(DType::F32)?.mean(D::Minus1)?;
        let ours: Vec<f32> = y.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_reduction,
            &format!("mean {:?}", cfg.dtype),
        );
    }
    Ok(())
}

// == Comparison ==

#[test]
fn comparisons() -> Result<()> {
    let t1 = Tensor::from_vec(vec![0u32, 1, 2, 3, 4, 5], (3, 2), dev())?;
    let t2 = Tensor::from_vec(vec![1u32, 0, 3, 3, 4, 7], (3, 2), dev())?;
    assert_eq!(
        t1.eq_t(&t2)?.to_vec2::<u8>()?,
        vec![vec![0, 0], vec![0, 1], vec![1, 0]]
    );
    assert_eq!(
        t1.ne_t(&t2)?.to_vec2::<u8>()?,
        vec![vec![1, 1], vec![1, 0], vec![0, 1]]
    );
    assert_eq!(
        t1.lt_t(&t2)?.to_vec2::<u8>()?,
        vec![vec![1, 0], vec![1, 0], vec![0, 1]]
    );
    assert_eq!(
        t1.gt_t(&t2)?.to_vec2::<u8>()?,
        vec![vec![0, 1], vec![0, 0], vec![0, 0]]
    );
    assert_eq!(
        t1.le_t(&t2)?.to_vec2::<u8>()?,
        vec![vec![1, 0], vec![1, 1], vec![1, 1]]
    );
    assert_eq!(
        t1.ge_t(&t2)?.to_vec2::<u8>()?,
        vec![vec![0, 1], vec![0, 1], vec![1, 0]]
    );
    Ok(())
}

// == Where ==

#[test]
fn where_cond() -> Result<()> {
    let cond = Tensor::from_vec(
        vec![0u8, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        (2, 5),
        dev(),
    )?;
    let a = Tensor::from_vec(
        vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9.],
        (2, 5),
        dev(),
    )?;
    let b = Tensor::from_vec(
        vec![10f32, 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        (2, 5),
        dev(),
    )?;
    let result = cond.where_cond(&a, &b)?;
    let v: Vec<f32> = result.flatten_all()?.to_vec1()?;
    assert_eq!(v, vec![10., 1., 12., 3., 14., 5., 6., 7., 18., 19.]);
    Ok(())
}

// == Index operations ==

#[test]
fn index_select() -> Result<()> {
    let ids = Tensor::from_vec(vec![0u32, 2, 1], 3, dev())?;
    let t = Tensor::from_vec(
        vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        (4, 3),
        dev(),
    )?;
    let r = t.index_select(&ids, 1)?;
    assert_eq!(
        r.to_vec2::<f32>()?,
        vec![
            vec![0., 2., 1.],
            vec![3., 5., 4.],
            vec![6., 8., 7.],
            vec![9., 11., 10.],
        ]
    );
    let r = t.index_select(&ids, 0)?;
    assert_eq!(
        r.to_vec2::<f32>()?,
        vec![vec![0., 1., 2.], vec![6., 7., 8.], vec![3., 4., 5.],]
    );
    Ok(())
}

#[test]
fn gather() -> Result<()> {
    let t = Tensor::from_vec(
        vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        (4, 3),
        dev(),
    )?;
    let ids = Tensor::from_vec(vec![0u32, 2, 1, 0], (4, 1), dev())?;
    let r = t.gather(&ids, 1)?;
    assert_eq!(
        r.to_vec2::<f32>()?,
        vec![vec![0.], vec![5.], vec![7.], vec![9.]]
    );
    Ok(())
}

#[test]
fn scatter_add() -> Result<()> {
    let init = Tensor::ones((4, 5), DType::F32, dev())?;
    let ids = Tensor::from_vec(
        vec![0u32, 1, 2, 3, 4, 0, 3, 3, 1, 2, 0, 4],
        (4, 3),
        dev(),
    )?;
    let src = Tensor::from_vec(
        vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        (4, 3),
        dev(),
    )?;
    let r = init.scatter_add(&ids, &src, 1)?;
    assert_eq!(
        r.to_vec2::<f32>()?,
        vec![
            vec![1., 2., 3., 1., 1.],
            vec![6., 1., 1., 4., 5.],
            vec![1., 9., 1., 14., 1.],
            vec![11., 1., 10., 1., 12.],
        ]
    );
    Ok(())
}

// == Transpose ==

#[test]
fn transpose() -> Result<()> {
    let t = Tensor::from_vec(
        vec![3f32, 1., 4., 1., 5., 2., 1., 7., 8., 2.],
        (2, 5),
        dev(),
    )?;
    let tt = t.t()?;
    assert_eq!(tt.dims(), &[5, 2]);
    assert_eq!(
        tt.to_vec2::<f32>()?,
        vec![
            vec![3., 2.],
            vec![1., 1.],
            vec![4., 7.],
            vec![1., 8.],
            vec![5., 2.],
        ]
    );
    Ok(())
}

// == Type casting ==

#[test]
fn dtype_cast() -> Result<()> {
    let t = Tensor::from_vec(vec![1.5f32, 2.7, -0.3], 3, dev())?;
    let bf16 = t.to_dtype(DType::BF16)?;
    let back = bf16.to_dtype(DType::F32)?;
    let v: Vec<f32> = back.to_vec1()?;
    for (&orig, &cast) in [1.5f32, 2.7, -0.3].iter().zip(v.iter()) {
        assert!(
            (orig - cast).abs() < 0.05,
            "BF16 roundtrip: {orig} != {cast}"
        );
    }
    Ok(())
}

// == Affine ==

#[test]
fn affine() -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], 4, dev())?;
    let r = t.affine(2.0, 1.0)?; // 2*x + 1
    assert_eq!(r.to_vec1::<f32>()?, vec![3.0, 5.0, 7.0, 9.0]);
    Ok(())
}

// == Softmax ==

#[test]
fn softmax() -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3.], 3, dev())?;
    let s = t.softmax(0)?;
    let v = common::to_vec1_round(&s, 4)?;
    assert_eq!(v, vec![0.0900, 0.2447, 0.6652]);
    let sum: f32 = s.to_vec1::<f32>()?.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn softmax_vs_pytorch() -> Result<()> {
    let len = 2048;
    let data = common::pseudo_random(len, 5.0);

    for cfg in common::ALL_DTYPES {
        let reference = require_pytorch_ref!(
            &[("x", &data)],
            &format!(r#"
x = read_input("x", {py}).reshape(1, {len})
y = torch.nn.functional.softmax(x, dim=-1).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let x = Tensor::from_vec(data.clone(), (1, len), dev())?.to_dtype(cfg.dtype)?;
        let y = x.softmax(D::Minus1)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_reduction,
            &format!("softmax {:?}", cfg.dtype),
        );

        // Verify sum ~ 1.0 (f32 only for tight tolerance)
        if cfg.dtype == DType::F32 {
            let sum: f32 = ours.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
        }
    }
    Ok(())
}

#[test]
fn softmax_extreme_values_vs_pytorch() -> Result<()> {
    // Mix of large positive, large negative, and normal values
    let mut data = common::pseudo_random(512, 6.0);
    data[0] = 100.0;
    data[1] = -100.0;
    data[10] = 50.0;
    data[11] = -50.0;

    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(1, 512)
y = torch.nn.functional.softmax(x, dim=-1)
write_output(y)
"#
    );

    let x = Tensor::from_vec(data, (1, 512), dev())?;
    let y = x.softmax(D::Minus1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-6, "softmax_extreme");
    Ok(())
}

// == Embedding ==

#[test]
fn embedding() -> Result<()> {
    let ids = Tensor::from_vec(vec![0u32, 2, 1], 3, dev())?;
    let table = Tensor::from_vec(vec![0f32, 1., 2., 3., 4., 5.], (3, 2), dev())?;
    let result = table.embedding(&ids)?;
    assert_eq!(
        result.to_vec2::<f32>()?,
        vec![vec![0., 1.], vec![4., 5.], vec![2., 3.]]
    );
    Ok(())
}

// == Cat ==

#[test]
fn cat_dim0() -> Result<()> {
    let a = Tensor::from_vec(vec![1f32, 2., 3.], (1, 3), dev())?;
    let b = Tensor::from_vec(vec![4f32, 5., 6.], (1, 3), dev())?;
    let c = Tensor::cat(&[&a, &b], 0)?;
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![vec![1., 2., 3.], vec![4., 5., 6.]]
    );
    Ok(())
}

#[test]
fn cat_dim1() -> Result<()> {
    let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), dev())?;
    let b = Tensor::from_vec(vec![5f32, 6., 7., 8.], (2, 2), dev())?;
    let c = Tensor::cat(&[&a, &b], 1)?;
    assert_eq!(c.dims(), &[2, 4]);
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![vec![1., 2., 5., 6.], vec![3., 4., 7., 8.]]
    );
    Ok(())
}

// == Non-contiguous tensor operations ==

#[test]
fn transpose_matmul() -> Result<()> {
    let a = Tensor::from_vec(
        (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        (3, 4),
        dev(),
    )?;
    let b = Tensor::from_vec(
        (0..8).map(|i| i as f32).collect::<Vec<_>>(),
        (4, 2),
        dev(),
    )?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![vec![28., 34.], vec![76., 98.], vec![124., 162.]]
    );

    let at = a.t()?; // (4, 3)
    assert!(!at.is_contiguous());
    let b2 = Tensor::from_vec(
        (0..6).map(|i| i as f32).collect::<Vec<_>>(),
        (3, 2),
        dev(),
    )?;
    let c2 = at.matmul(&b2)?;
    assert_eq!(c2.dims(), &[4, 2]);
    assert_eq!(c2.to_vec2::<f32>()?[0], vec![40., 52.]);
    Ok(())
}

#[test]
fn narrow_then_ops() -> Result<()> {
    let t = Tensor::from_vec(
        (0..20).map(|i| i as f32).collect::<Vec<_>>(),
        (4, 5),
        dev(),
    )?;
    let n = t.narrow(1, 1, 3)?; // cols 1..4, shape (4, 3)
    assert_eq!(n.dims(), &[4, 3]);

    let s = n.sum_all()?;
    assert_eq!(s.to_scalar::<f32>()?, 114.0);

    let w = Tensor::ones((3, 2), DType::F32, dev())?;
    let r = n.matmul(&w)?;
    assert_eq!(r.dims(), &[4, 2]);
    assert_eq!(r.to_vec2::<f32>()?[0], vec![6., 6.]);
    Ok(())
}

#[test]
fn non_contiguous_reduction() -> Result<()> {
    let t = Tensor::from_vec(
        (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        (3, 4),
        dev(),
    )?;
    let tt = t.t()?; // (4, 3), non-contiguous
    let s0 = tt.sum_keepdim(0)?;
    assert_eq!(s0.dims(), &[1, 3]);
    assert_eq!(s0.to_vec2::<f32>()?, vec![vec![6., 22., 38.]]);
    Ok(())
}

// == Broadcasting edge cases ==

#[test]
fn broadcast_add_asymmetric() -> Result<()> {
    let a = Tensor::from_vec(vec![1f32, 2., 3.], (3, 1), dev())?;
    let b = Tensor::from_vec(vec![10f32, 20., 30., 40.], (1, 4), dev())?;
    let c = a.broadcast_add(&b)?;
    assert_eq!(c.dims(), &[3, 4]);
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![
            vec![11., 21., 31., 41.],
            vec![12., 22., 32., 42.],
            vec![13., 23., 33., 43.],
        ]
    );
    Ok(())
}

#[test]
fn broadcast_mul_with_scalar_dim() -> Result<()> {
    let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), dev())?;
    let s = Tensor::from_vec(vec![0.5f32], 1, dev())?;
    let c = a.broadcast_mul(&s.reshape((1, 1))?)?;
    assert_eq!(
        common::to_vec2_round(&c, 1)?,
        vec![vec![0.5, 1.0, 1.5], vec![2.0, 2.5, 3.0]]
    );
    Ok(())
}

// == Shape manipulation ==

#[test]
fn squeeze_unsqueeze() -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3.], 3, dev())?;
    assert_eq!(t.dims(), &[3]);
    let u = t.unsqueeze(0)?;
    assert_eq!(u.dims(), &[1, 3]);
    let u2 = t.unsqueeze(1)?;
    assert_eq!(u2.dims(), &[3, 1]);
    let s = u.squeeze(0)?;
    assert_eq!(s.dims(), &[3]);
    assert_eq!(s.to_vec1::<f32>()?, vec![1., 2., 3.]);
    Ok(())
}

#[test]
fn reshape_non_contiguous() -> Result<()> {
    let t = Tensor::from_vec(
        (0..6).map(|i| i as f32).collect::<Vec<_>>(),
        (2, 3),
        dev(),
    )?;
    let tt = t.t()?; // (3, 2), non-contiguous
    let r = tt.reshape((6,))?;
    assert_eq!(r.to_vec1::<f32>()?, vec![0., 3., 1., 4., 2., 5.]);
    Ok(())
}

#[test]
fn flatten() -> Result<()> {
    let t = Tensor::from_vec(
        (0..24).map(|i| i as f32).collect::<Vec<_>>(),
        (2, 3, 4),
        dev(),
    )?;
    let f = t.flatten_all()?;
    assert_eq!(f.dims(), &[24]);
    assert_eq!(f.to_vec1::<f32>()?[0], 0.);
    assert_eq!(f.to_vec1::<f32>()?[23], 23.);

    let f2 = t.flatten(0, 1)?;
    assert_eq!(f2.dims(), &[6, 4]);
    Ok(())
}

// == Comparison operators ==

#[test]
fn comparison_ops() -> Result<()> {
    let a = Tensor::from_vec(vec![0f32, 1., 2., 3., 4., 5.], (2, 3), dev())?;
    let b = Tensor::from_vec(vec![1f32, 0., 3., 3., 4., 7.], (2, 3), dev())?;

    let eq = a.eq_t(&b)?;
    assert_eq!(eq.to_vec2::<u8>()?, vec![vec![0, 0, 0], vec![1, 1, 0]]);

    let lt = a.lt_t(&b)?;
    assert_eq!(lt.to_vec2::<u8>()?, vec![vec![1, 0, 1], vec![0, 0, 1]]);

    let gt = a.gt_t(&b)?;
    assert_eq!(gt.to_vec2::<u8>()?, vec![vec![0, 1, 0], vec![0, 0, 0]]);

    let le = a.le_t(&b)?;
    assert_eq!(le.to_vec2::<u8>()?, vec![vec![1, 0, 1], vec![1, 1, 1]]);

    let ge = a.ge_t(&b)?;
    assert_eq!(ge.to_vec2::<u8>()?, vec![vec![0, 1, 0], vec![1, 1, 0]]);
    Ok(())
}

// == Gather with multiple index dims ==

#[test]
fn gather_2d_indices() -> Result<()> {
    let t = Tensor::from_vec(
        (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        (4, 3),
        dev(),
    )?;
    let ids = Tensor::from_vec(vec![0u32, 0, 2, 0, 1, 1, 0, 2], (4, 2), dev())?;
    let r = t.gather(&ids, 1)?;
    assert_eq!(r.dims(), &[4, 2]);
    assert_eq!(
        r.to_vec2::<f32>()?,
        vec![vec![0., 0.], vec![5., 3.], vec![7., 7.], vec![9., 11.],]
    );
    Ok(())
}

// == Numerical precision ==

#[test]
fn softmax_large_values() -> Result<()> {
    let t = Tensor::from_vec(vec![1000f32, 1001., 1002.], 3, dev())?;
    let s = t.softmax(D::Minus1)?;
    let v = common::to_vec1_round(&s, 4)?;
    assert_eq!(v, vec![0.0900, 0.2447, 0.6652]);
    let sum: f32 = s.to_vec1::<f32>()?.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn sum_large_reduction() -> Result<()> {
    let n = 10000usize;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let t = Tensor::from_vec(data.clone(), n, dev())?;
    let s = t.sum_all()?.to_scalar::<f32>()?;
    let expected: f32 = data.iter().sum();
    assert!((s - expected).abs() < 0.1, "sum diff: {} vs {}", s, expected);
    Ok(())
}

// == Chunk / split ==

#[test]
fn chunk() -> Result<()> {
    let t = Tensor::from_vec(
        (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        (4, 3),
        dev(),
    )?;
    let chunks = t.chunk(2, 0)?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[2, 3]);
    assert_eq!(chunks[1].dims(), &[2, 3]);
    assert_eq!(
        chunks[0].to_vec2::<f32>()?,
        vec![vec![0., 1., 2.], vec![3., 4., 5.]]
    );
    assert_eq!(
        chunks[1].to_vec2::<f32>()?,
        vec![vec![6., 7., 8.], vec![9., 10., 11.]]
    );
    Ok(())
}

// == Where-cond (2D) ==

#[test]
fn where_cond_2d() -> Result<()> {
    let cond = Tensor::from_vec(vec![1u8, 0, 1, 0], (2, 2), dev())?;
    let on_true = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), dev())?;
    let on_false = Tensor::from_vec(vec![10f32, 20., 30., 40.], (2, 2), dev())?;
    let r = cond.where_cond(&on_true, &on_false)?;
    assert_eq!(r.to_vec2::<f32>()?, vec![vec![1., 20.], vec![3., 40.]]);
    Ok(())
}

// == Contiguous ==

#[test]
fn contiguous_is_noop_when_already_contiguous() -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3.], 3, dev())?;
    assert!(t.is_contiguous());
    let c = t.contiguous()?;
    assert!(c.is_contiguous());
    assert_eq!(c.to_vec1::<f32>()?, vec![1., 2., 3.]);
    Ok(())
}

#[test]
fn contiguous_fixes_transpose() -> Result<()> {
    let t = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), dev())?;
    let tt = t.t()?;
    assert!(!tt.is_contiguous());
    let c = tt.contiguous()?;
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(
        c.to_vec2::<f32>()?,
        vec![vec![1., 4.], vec![2., 5.], vec![3., 6.]]
    );
    Ok(())
}

// == PyTorch-referenced op tests ==

#[test]
fn broadcast_div_vs_pytorch() -> Result<()> {
    let a_data = vec![6f32, 12., 18., 24., 30., 36.];
    let b_data = vec![2f32, 3., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("a", &a_data), ("b", &b_data)],
        r#"
a = read_input("a").reshape(2, 3)
b = read_input("b").reshape(1, 3)
y = a / b
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 3);
    let a = Tensor::from_vec(a_data, (2, 3), dev())?;
    let b = Tensor::from_vec(b_data, (1, 3), dev())?;
    let y = a.broadcast_div(&b)?;
    let ours = y.to_vec2::<f32>()?;
    common::assert_close_2d(&ours, &reference, 1e-5, "broadcast_div");
    Ok(())
}

#[test]
fn neg_vs_pytorch() -> Result<()> {
    let data = vec![1f32, -2., 3., -4., 0., 5.5];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output(-x)
"#
    );
    let x = Tensor::from_vec(data, 6, dev())?;
    let y = x.neg()?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-6, "neg");
    Ok(())
}

#[test]
fn abs_vs_pytorch() -> Result<()> {
    let data = vec![-3f32, 1., -0.5, 0., 2.7, -1.8];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output(x.abs())
"#
    );
    let x = Tensor::from_vec(data, 6, dev())?;
    let y = x.abs()?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-6, "abs");
    Ok(())
}

#[test]
fn recip_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 4., 0.5, -2., 0.1];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output(x.reciprocal())
"#
    );
    let x = Tensor::from_vec(data, 6, dev())?;
    let y = x.recip()?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-5, "recip");
    Ok(())
}

#[test]
fn powf_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 0.5];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output(x.pow(2.0))
"#
    );
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.powf(2.0)?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-5, "powf(2.0)");
    Ok(())
}

#[test]
fn powf_fractional_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 4., 9., 16., 25.];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output(x.pow(0.5))
"#
    );
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.powf(0.5)?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-5, "powf(0.5)");
    Ok(())
}

#[test]
fn gelu_erf_vs_pytorch() -> Result<()> {
    let data = vec![-3f32, -1., 0., 0.5, 1., 2.7];

    for cfg in common::ALL_DTYPES {
        let reference = require_pytorch_ref!(
            &[("x", &data)],
            &format!(r#"
x = read_input("x", {py})
y = torch.nn.functional.gelu(x, approximate='none').float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );
        let x = Tensor::from_vec(data.clone(), 6, dev())?.to_dtype(cfg.dtype)?;
        let y = x.gelu_erf()?.to_dtype(DType::F32)?;
        common::assert_close(
            &y.to_vec1::<f32>()?,
            &reference,
            cfg.atol_default,
            &format!("gelu_erf {:?}", cfg.dtype),
        );
    }
    Ok(())
}

// == mean / mean_all vs PyTorch ==

#[test]
fn mean_small_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6.];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 3)
write_output(x.mean(dim=1))
"#
    );
    let x = Tensor::from_vec(data, (2, 3), dev())?;
    let y = x.mean(1)?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-5, "mean(dim=1)");
    Ok(())
}

#[test]
fn mean_dim0_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6.];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 3)
write_output(x.mean(dim=0))
"#
    );
    let x = Tensor::from_vec(data, (2, 3), dev())?;
    let y = x.mean(0)?;
    common::assert_close(&y.to_vec1::<f32>()?, &reference, 1e-5, "mean(dim=0)");
    Ok(())
}

#[test]
fn mean_all_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 3)
write_output(x.mean().reshape(1))
"#
    );
    let reference: f32 = ref_flat[0];
    let x = Tensor::from_vec(data, (2, 3), dev())?;
    let y = x.mean_all()?;
    let ours = y.to_scalar::<f32>()?;
    assert!(
        (ours - reference).abs() < 1e-5,
        "mean_all: ours={ours}, pytorch={reference}"
    );
    Ok(())
}

// == ne_t vs PyTorch ==

#[test]
fn ne_t_vs_pytorch() -> Result<()> {
    let a_data = vec![0f32, 1., 2., 3., 4., 5.];
    let b_data = vec![1f32, 1., 3., 3., 4., 7.];
    let ref_flat = require_pytorch_ref!(
        &[("a", &a_data), ("b", &b_data)],
        r#"
a = read_input("a")
b = read_input("b")
write_output((a != b).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let a = Tensor::from_vec(a_data, 6, dev())?;
    let b = Tensor::from_vec(b_data, 6, dev())?;
    let y = a.ne_t(&b)?;
    let ours = y.to_vec1::<u8>()?;
    assert_eq!(ours, reference, "ne_t mismatch");
    Ok(())
}

// == scalar comparisons vs PyTorch ==

#[test]
fn scalar_ge_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output((x >= 3.0).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.ge(3.0f32)?;
    assert_eq!(y.to_vec1::<u8>()?, reference, "ge(3.0)");
    Ok(())
}

#[test]
fn scalar_gt_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output((x > 3.0).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.gt(3.0f32)?;
    assert_eq!(y.to_vec1::<u8>()?, reference, "gt(3.0)");
    Ok(())
}

#[test]
fn scalar_le_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output((x <= 3.0).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.le(3.0f32)?;
    assert_eq!(y.to_vec1::<u8>()?, reference, "le(3.0)");
    Ok(())
}

#[test]
fn scalar_lt_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output((x < 3.0).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.lt(3.0f32)?;
    assert_eq!(y.to_vec1::<u8>()?, reference, "lt(3.0)");
    Ok(())
}

#[test]
fn scalar_eq_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output((x == 3.0).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.eq_scalar(3.0f32)?;
    assert_eq!(y.to_vec1::<u8>()?, reference, "eq_scalar(3.0)");
    Ok(())
}

#[test]
fn scalar_ne_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
write_output((x != 3.0).float())
"#
    );
    let reference: Vec<u8> = ref_flat.iter().map(|&v| v as u8).collect();
    let x = Tensor::from_vec(data, 5, dev())?;
    let y = x.ne_scalar(3.0f32)?;
    assert_eq!(y.to_vec1::<u8>()?, reference, "ne_scalar(3.0)");
    Ok(())
}

// == index_add vs PyTorch ==

#[test]
fn index_add_vs_pytorch() -> Result<()> {
    let base_data = vec![0f32; 12];
    let ids_data = vec![1u32, 3];
    let src_data = vec![10f32, 20., 30., 40., 50., 60.];
    let ref_flat = require_pytorch_ref!(
        &[("src", &src_data)],
        r#"
base = torch.zeros(4, 3)
ids = torch.tensor([1, 3], dtype=torch.long)
src = read_input("src").reshape(2, 3)
result = base.index_add(0, ids, src)
write_output(result)
"#
    );
    let reference = common::unflatten(&ref_flat, 3);
    let base = Tensor::from_vec(base_data, (4, 3), dev())?;
    let ids = Tensor::from_vec(ids_data, 2, dev())?;
    let src = Tensor::from_vec(src_data, (2, 3), dev())?;
    let y = base.index_add(&ids, &src, 0)?;
    let ours = y.to_vec2::<f32>()?;
    common::assert_close_2d(&ours, &reference, 1e-5, "index_add");
    Ok(())
}

// == stack vs PyTorch ==

#[test]
fn stack_dim0_vs_pytorch() -> Result<()> {
    let a_data = vec![1f32, 2., 3.];
    let b_data = vec![4f32, 5., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("a", &a_data), ("b", &b_data)],
        r#"
a = read_input("a")
b = read_input("b")
write_output(torch.stack([a, b], dim=0))
"#
    );
    let reference = common::unflatten(&ref_flat, 3);
    let a = Tensor::from_vec(a_data, 3, dev())?;
    let b = Tensor::from_vec(b_data, 3, dev())?;
    let y = Tensor::stack(&[&a, &b], 0)?;
    assert_eq!(y.dims(), &[2, 3]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "stack_dim0");
    Ok(())
}

#[test]
fn stack_dim1_vs_pytorch() -> Result<()> {
    let a_data = vec![1f32, 2., 3.];
    let b_data = vec![4f32, 5., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("a", &a_data), ("b", &b_data)],
        r#"
a = read_input("a")
b = read_input("b")
write_output(torch.stack([a, b], dim=1))
"#
    );
    let reference = common::unflatten(&ref_flat, 2);
    let a = Tensor::from_vec(a_data, 3, dev())?;
    let b = Tensor::from_vec(b_data, 3, dev())?;
    let y = Tensor::stack(&[&a, &b], 1)?;
    assert_eq!(y.dims(), &[3, 2]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "stack_dim1");
    Ok(())
}

// == pad_with_zeros vs PyTorch ==

#[test]
fn pad_with_zeros_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 3)
y = torch.nn.functional.pad(x, (2, 1))
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 6);
    let x = Tensor::from_vec(data, (2, 3), dev())?;
    let y = x.pad_with_zeros(1, 2, 1)?;
    assert_eq!(y.dims(), &[2, 6]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "pad_with_zeros");
    Ok(())
}

#[test]
fn pad_with_zeros_dim0_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 3)
y = torch.nn.functional.pad(x, (0, 0, 1, 2))
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 3);
    let x = Tensor::from_vec(data, (2, 3), dev())?;
    let y = x.pad_with_zeros(0, 1, 2)?;
    assert_eq!(y.dims(), &[5, 3]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "pad_with_zeros_dim0");
    Ok(())
}

// == repeat vs PyTorch ==

#[test]
fn repeat_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 3)
y = x.repeat(2, 3)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 9);
    let x = Tensor::from_vec(data, (2, 3), dev())?;
    let y = x.repeat(&[2, 3])?;
    assert_eq!(y.dims(), &[4, 9]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "repeat");
    Ok(())
}

// == broadcast_as vs PyTorch ==

#[test]
fn broadcast_as_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(1, 4)
y = x.expand(3, 4)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 4);
    let x = Tensor::from_vec(data, (1, 4), dev())?;
    let y = x.broadcast_as((3, 4))?;
    assert_eq!(y.dims(), &[3, 4]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "broadcast_as");
    Ok(())
}

#[test]
fn broadcast_as_column_vs_pytorch() -> Result<()> {
    let data = vec![10f32, 20., 30.];
    let ref_flat = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(3, 1)
y = x.expand(3, 4)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 4);
    let x = Tensor::from_vec(data, (3, 1), dev())?;
    let y = x.broadcast_as((3, 4))?;
    assert_eq!(y.dims(), &[3, 4]);
    common::assert_close_2d(&y.to_vec2::<f32>()?, &reference, 1e-6, "broadcast_as_col");
    Ok(())
}

// == sort_last_dim vs PyTorch ==

#[test]
fn sort_last_dim_asc_vs_pytorch() -> Result<()> {
    let data = vec![3f32, 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.];
    let result = require_pytorch_ref_multi!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 6)
vals, idxs = torch.sort(x, dim=-1)
write_outputs(vals=vals, idxs=idxs)
"#
    );
    let ref_vals = common::unflatten(&result["vals"], 6);
    let ref_idxs: Vec<Vec<u32>> = result["idxs"].iter().map(|&v| v as u32).collect::<Vec<u32>>()
        .chunks(6).map(|c| c.to_vec()).collect();

    let x = Tensor::from_vec(data, (2, 6), dev())?;
    let (vals, idxs) = x.sort_last_dim(true)?;
    let our_vals = vals.to_vec2::<f32>()?;
    let our_idxs = idxs.to_vec2::<u32>()?;
    common::assert_close_2d(&our_vals, &ref_vals, 1e-6, "sort_asc_vals");
    assert_eq!(our_idxs, ref_idxs, "sort_asc_indices mismatch");
    Ok(())
}

#[test]
fn sort_last_dim_desc_vs_pytorch() -> Result<()> {
    let data = vec![3f32, 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.];
    let result = require_pytorch_ref_multi!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(2, 6)
vals, idxs = torch.sort(x, dim=-1, descending=True)
write_outputs(vals=vals, idxs=idxs)
"#
    );
    let ref_vals = common::unflatten(&result["vals"], 6);
    let ref_idxs: Vec<Vec<u32>> = result["idxs"].iter().map(|&v| v as u32).collect::<Vec<u32>>()
        .chunks(6).map(|c| c.to_vec()).collect();

    let x = Tensor::from_vec(data, (2, 6), dev())?;
    let (vals, idxs) = x.sort_last_dim(false)?;
    let our_vals = vals.to_vec2::<f32>()?;
    let our_idxs = idxs.to_vec2::<u32>()?;
    common::assert_close_2d(&our_vals, &ref_vals, 1e-6, "sort_desc_vals");
    assert_eq!(our_idxs, ref_idxs, "sort_desc_indices mismatch");
    Ok(())
}

// == RoPE THD ==

#[test]
fn rope_thd_vs_pytorch() -> Result<()> {
    let b = 1;
    let l = 2;
    let h = 2;
    let d = 4;

    let x_data: Vec<f32> = (0..b * l * h * d).map(|i| (i as f32) * 0.1).collect();
    let cos_data: Vec<f32> = (0..l * d / 2).map(|i| ((i as f32) * 0.3).cos()).collect();
    let sin_data: Vec<f32> = (0..l * d / 2).map(|i| ((i as f32) * 0.3).sin()).collect();

    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("cos", &cos_data), ("sin", &sin_data)],
        &format!(r#"
x = read_input("x").reshape({b}, {l}, {h}, {d})
cos = read_input("cos").reshape({l}, {half_d})
sin = read_input("sin").reshape({l}, {half_d})
cat_cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2)
cat_sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2)
x1 = x[..., :x.shape[-1]//2]
x2 = x[..., x.shape[-1]//2:]
rotated = torch.cat([-x2, x1], dim=-1)
y = x * cat_cos + rotated * cat_sin
write_output(y)
"#,
            half_d = d / 2
        )
    );

    let x = Tensor::from_vec(x_data, (b, l, h, d), dev())?;
    let cos = Tensor::from_vec(cos_data, (l, d / 2), dev())?;
    let sin = Tensor::from_vec(sin_data, (l, d / 2), dev())?;
    let y = x.rope_thd(&cos, &sin)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "rope_thd");
    Ok(())
}

// == Conv1D vs PyTorch ==

#[test]
fn conv1d_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 2., 3., 4., 5.];
    let w_data = vec![1f32, 0., -1.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 5)
w = read_input("w").reshape(1, 1, 3)
y = torch.nn.functional.conv1d(x, w, padding=0, stride=1)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 5), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3), dev())?;
    let y = x.conv1d(&w, 0, 1, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv1d");
    Ok(())
}

#[test]
fn conv1d_padded_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 2., 3., 4., 5.];
    let w_data = vec![1f32, 0., -1.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 5)
w = read_input("w").reshape(1, 1, 3)
y = torch.nn.functional.conv1d(x, w, padding=1, stride=1)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 5), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3), dev())?;
    let y = x.conv1d(&w, 1, 1, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv1d_padded");
    Ok(())
}

#[test]
fn conv1d_stride2_vs_pytorch() -> Result<()> {
    let x_data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let w_data = vec![1f32, 2., 3.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 8)
w = read_input("w").reshape(1, 1, 3)
y = torch.nn.functional.conv1d(x, w, padding=0, stride=2)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 8), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3), dev())?;
    let y = x.conv1d(&w, 0, 2, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv1d_stride2");
    Ok(())
}

#[test]
fn conv1d_multi_channel_vs_pytorch() -> Result<()> {
    let x_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let w_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 2, 6)
w = read_input("w").reshape(2, 2, 3)
y = torch.nn.functional.conv1d(x, w, padding=0, stride=1)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 2, 6), dev())?;
    let w = Tensor::from_vec(w_data, (2, 2, 3), dev())?;
    let y = x.conv1d(&w, 0, 1, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-4, "conv1d_multi_channel");
    Ok(())
}

// == Conv2D vs PyTorch ==

#[test]
fn conv2d_vs_pytorch() -> Result<()> {
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let w_data: Vec<f32> = vec![1., 0., -1., 2., 0., -2., 1., 0., -1.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 4, 4)
w = read_input("w").reshape(1, 1, 3, 3)
y = torch.nn.functional.conv2d(x, w, padding=0, stride=1)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 4, 4), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3, 3), dev())?;
    let y = x.conv2d(&w, 0, 1, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv2d");
    Ok(())
}

#[test]
fn conv2d_padded_vs_pytorch() -> Result<()> {
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let w_data: Vec<f32> = vec![1., 0., -1., 2., 0., -2., 1., 0., -1.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 4, 4)
w = read_input("w").reshape(1, 1, 3, 3)
y = torch.nn.functional.conv2d(x, w, padding=1, stride=1)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 4, 4), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3, 3), dev())?;
    let y = x.conv2d(&w, 1, 1, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv2d_padded");
    Ok(())
}

// == Conv Transpose1D vs PyTorch ==

#[test]
fn conv_transpose1d_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 2., 3.];
    let w_data = vec![1f32, 0., -1.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 3)
w = read_input("w").reshape(1, 1, 3)
y = torch.nn.functional.conv_transpose1d(x, w, padding=0, stride=1)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 3), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3), dev())?;
    let y = x.conv_transpose1d(&w, 0, 0, 1, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv_transpose1d");
    Ok(())
}

#[test]
fn conv_transpose1d_stride2_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 2., 3.];
    let w_data = vec![1f32, 2., 3.];
    let reference = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 1, 3)
w = read_input("w").reshape(1, 1, 3)
y = torch.nn.functional.conv_transpose1d(x, w, padding=0, stride=2)
write_output(y)
"#
    );
    let x = Tensor::from_vec(x_data, (1, 1, 3), dev())?;
    let w = Tensor::from_vec(w_data, (1, 1, 3), dev())?;
    let y = x.conv_transpose1d(&w, 0, 0, 2, 1, 1)?;
    let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-5, "conv_transpose1d_stride2");
    Ok(())
}

// == Interpolate1d ==

#[test]
fn interpolate1d_upsample_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4.];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(1, 1, 4)
y = torch.nn.functional.interpolate(x, size=8, mode='nearest')
write_output(y)
"#
    );
    let x = Tensor::from_vec(data, (1, 1, 4), dev())?;
    let y = x.interpolate1d(8)?;
    common::assert_close(
        &y.flatten_all()?.to_vec1::<f32>()?,
        &reference,
        1e-6,
        "interpolate1d_up",
    );
    Ok(())
}

#[test]
fn interpolate1d_downsample_vs_pytorch() -> Result<()> {
    let data = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(1, 1, 8)
y = torch.nn.functional.interpolate(x, size=3, mode='nearest')
write_output(y)
"#
    );
    let x = Tensor::from_vec(data, (1, 1, 8), dev())?;
    let y = x.interpolate1d(3)?;
    common::assert_close(
        &y.flatten_all()?.to_vec1::<f32>()?,
        &reference,
        1e-6,
        "interpolate1d_down",
    );
    Ok(())
}

#[test]
fn interpolate1d_multichannel_vs_pytorch() -> Result<()> {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(1, 3, 4)
y = torch.nn.functional.interpolate(x, size=6, mode='nearest')
write_output(y)
"#
    );
    let x = Tensor::from_vec(data, (1, 3, 4), dev())?;
    let y = x.interpolate1d(6)?;
    common::assert_close(
        &y.flatten_all()?.to_vec1::<f32>()?,
        &reference,
        1e-6,
        "interpolate1d_multi",
    );
    Ok(())
}

// == Precision tests (merged from precision.rs) ==
// These test at realistic LLM dimensions with random data across multiple dtypes.

#[test]
fn rmsnorm_vs_pytorch() -> Result<()> {
    let batch = 8;
    let hidden = 1024;
    let x_data = common::pseudo_random(batch * hidden, 7.0);
    let w_data = common::pseudo_random(hidden, 8.0);
    let eps = 1e-6;

    for cfg in common::ALL_DTYPES {
        let reference = require_pytorch_ref!(
            &[("x", &x_data), ("w", &w_data)],
            &format!(r#"
x = read_input("x", {py}).reshape({batch}, {hidden})
w = read_input("w", {py})
y = torch.nn.functional.rms_norm(x.float(), ({hidden},), w.float(), {eps}).to({py}).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let x =
            Tensor::from_vec(x_data.clone(), (batch, hidden), dev())?.to_dtype(cfg.dtype)?;
        let w = Tensor::from_vec(w_data.clone(), (hidden,), dev())?.to_dtype(cfg.dtype)?;

        let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(w, eps);
        let y = norm.forward(&x)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_reduction,
            &format!("rmsnorm {:?}", cfg.dtype),
        );
    }
    Ok(())
}

use prelude_core::tensor::Module;

#[test]
fn linear_vs_pytorch() -> Result<()> {
    let batch = 16;
    let in_dim = 512;
    let out_dim = 256;
    let x_data = common::pseudo_random(batch * in_dim, 9.0);
    let w_data = common::pseudo_random(out_dim * in_dim, 10.0);

    for cfg in matmul_dtypes() {
        let reference = require_pytorch_ref!(
            &[("x", &x_data), ("w", &w_data)],
            &format!(r#"
x = read_input("x", {py}).reshape({batch}, {in_dim})
w = read_input("w", {py}).reshape({out_dim}, {in_dim})
y = (x @ w.T).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let x = Tensor::from_vec(x_data.clone(), (batch, in_dim), dev())?
            .to_dtype(cfg.dtype)?;
        let w = Tensor::from_vec(w_data.clone(), (out_dim, in_dim), dev())?
            .to_dtype(cfg.dtype)?;
        let linear = prelude_core::models::commons::linear::Linear::from_weight(w, None)?;
        let y = linear
            .forward(
                &x,
                &prelude_core::models::commons::BatchState::no_lora(),
                prelude_core::ops::select_ops(dev()),
            )?
            .to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_matmul,
            &format!("linear {:?}", cfg.dtype),
        );
    }
    Ok(())
}

// == Chained operations (norm -> linear -> silu_mul) ==

#[test]
fn chained_norm_linear_silu_vs_pytorch() -> Result<()> {
    let batch = 4;
    let hidden = 256;
    let inter = 512;
    let x_data = common::pseudo_random(batch * hidden, 11.0);
    let norm_w = common::pseudo_random(hidden, 12.0);
    let gate_w = common::pseudo_random(inter * hidden, 13.0);
    let up_w = common::pseudo_random(inter * hidden, 14.0);
    let eps = 1e-6;

    for cfg in matmul_dtypes() {
        let reference = require_pytorch_ref!(
            &[("x", &x_data), ("nw", &norm_w), ("gw", &gate_w), ("uw", &up_w)],
            &format!(r#"
x = read_input("x", {py}).reshape({batch}, {hidden})
nw = read_input("nw", {py})
gw = read_input("gw", {py}).reshape({inter}, {hidden})
uw = read_input("uw", {py}).reshape({inter}, {hidden})
h = torch.nn.functional.rms_norm(x.float(), ({hidden},), nw.float(), {eps}).to({py})
gate = h @ gw.T
up = h @ uw.T
y = (torch.nn.functional.silu(gate.float()) * up.float()).to({py}).float()
write_output(y)
"#,
                py = cfg.py_dtype
            )
        );

        let x = Tensor::from_vec(x_data.clone(), (batch, hidden), dev())?
            .to_dtype(cfg.dtype)?;
        let nw =
            Tensor::from_vec(norm_w.clone(), (hidden,), dev())?.to_dtype(cfg.dtype)?;
        let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(nw.clone(), eps);
        let ops = prelude_core::ops::select_ops(dev());

        let h = ops.rms_norm(&x, &nw, eps as f32)?;
        let gw = Tensor::from_vec(gate_w.clone(), (inter, hidden), dev())?
            .to_dtype(cfg.dtype)?;
        let uw = Tensor::from_vec(up_w.clone(), (inter, hidden), dev())?
            .to_dtype(cfg.dtype)?;
        let gate = h.matmul(&gw.t()?)?;
        let up = h.matmul(&uw.t()?)?;
        let y = ops.silu_mul(&gate, &up)?.to_dtype(DType::F32)?;
        let ours: Vec<f32> = y.flatten_all()?.to_vec1()?;
        common::assert_close(
            &ours,
            &reference,
            cfg.atol_chained,
            &format!("chained {:?}", cfg.dtype),
        );
    }
    Ok(())
}

// == Add + RmsNorm chained (residual connection) ==

#[test]
fn add_rmsnorm_vs_pytorch() -> Result<()> {
    let batch = 16;
    let hidden = 512;
    let x_data = common::pseudo_random(batch * hidden, 17.0);
    let h_data = common::pseudo_random(batch * hidden, 18.0);
    let w_data = common::pseudo_random(hidden, 19.0);
    let eps = 1e-6;

    let result = require_pytorch_ref_multi!(
        &[("x", &x_data), ("h", &h_data), ("w", &w_data)],
        &format!(r#"
x = read_input("x").reshape({batch}, {hidden})
h = read_input("h").reshape({batch}, {hidden})
w = read_input("w")
residual = x + h
normed = torch.nn.functional.rms_norm(residual, ({hidden},), w, {eps})
write_outputs(residual=residual, normed=normed)
"#)
    );
    let ref_residual = &result["residual"];
    let ref_normed = &result["normed"];

    let x = Tensor::from_vec(x_data, (batch, hidden), dev())?;
    let h = Tensor::from_vec(h_data, (batch, hidden), dev())?;
    let w = Tensor::from_vec(w_data.clone(), (hidden,), dev())?;
    let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(w.clone(), eps);
    let ops = prelude_core::ops::select_ops(dev());

    let (residual, normed) =
        ops.add_rmsnorm(&x, &h, &w, eps as f32)?;
    let our_res: Vec<f32> = residual.flatten_all()?.to_vec1()?;
    let our_normed: Vec<f32> = normed.flatten_all()?.to_vec1()?;
    common::assert_close(&our_res, ref_residual, 1e-5, "add_rmsnorm residual");
    common::assert_close(&our_normed, ref_normed, 1e-4, "add_rmsnorm normed");
    Ok(())
}

// ── Precision regression tests (erf, cast, reduce, where_cond) ──────

#[test]
fn gelu_erf_precision_vs_pytorch() -> Result<()> {
    let data = common::pseudo_random(128, 42.0);
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x")
y = torch.nn.functional.gelu(x, approximate='none')
write_output(y)
"#
    );
    let x = Tensor::from_vec(data, 128, dev())?;
    let y = x.gelu_erf()?;
    let ours: Vec<f32> = y.to_vec1()?;
    common::assert_close(&ours, &reference, 1e-6, "gelu_erf precision");
    Ok(())
}

#[test]
fn reduce_sum_precision_f32() -> Result<()> {
    // Large reduction — catches per-step f64→f32 round-trip bugs.
    let n = 4096;
    let data = common::pseudo_random(n, 99.0);
    let reference = require_pytorch_ref!(
        &[("x", &data)],
        r#"
x = read_input("x").reshape(4, 1024)
write_output(x.sum(dim=1))
"#
    );
    let x = Tensor::from_vec(data, (4, 1024), dev())?;
    let s = x.sum_keepdim(1)?.flatten_all()?;
    let ours: Vec<f32> = s.to_vec1()?;
    // 2e-5: summing 1024 f32 values, accumulation order differs (our f64 vs PyTorch f32).
    common::assert_close(&ours, &reference, 2e-5, "reduce_sum_f32");
    Ok(())
}

#[test]
fn reduce_max_min_multirow() -> Result<()> {
    // Regression: max/min init must be per-slice, not global first element.
    let t = Tensor::from_vec(vec![1f32, 5.0, 3.0, 8.0, 2.0, 7.0], (3, 2), dev())?;
    let mx = t.max_keepdim(1)?;
    assert_eq!(mx.to_vec2::<f32>()?, vec![vec![5.0], vec![8.0], vec![7.0]]);
    let mn = t.min_keepdim(1)?;
    assert_eq!(mn.to_vec2::<f32>()?, vec![vec![1.0], vec![3.0], vec![2.0]]);
    // dim=0 reduction: per-column max/min over 3 rows
    // col0: [1,3,2] → max=3, min=1
    // col1: [5,8,7] → max=8, min=5
    let mx0 = t.max_keepdim(0)?;
    assert_eq!(mx0.to_vec2::<f32>()?, vec![vec![3.0, 8.0]]);
    let mn0 = t.min_keepdim(0)?;
    assert_eq!(mn0.to_vec2::<f32>()?, vec![vec![1.0, 5.0]]);
    Ok(())
}

#[test]
fn cast_precision() -> Result<()> {
    // f32 → bf16 → f32 round-trip
    let data = vec![1.5f32, -2.75, 0.0, 100.0];
    let t = Tensor::from_vec(data.clone(), 4, dev())?;
    let bf = t.to_dtype(DType::BF16)?;
    let back: Vec<f32> = bf.to_dtype(DType::F32)?.to_vec1()?;
    // These values are exactly representable in bf16, so exact match.
    assert_eq!(back, data);
    // f32 → f16 → f32 round-trip (these values are exactly representable in f16 too)
    let f16t = t.to_dtype(DType::F16)?;
    let back16: Vec<f32> = f16t.to_dtype(DType::F32)?.to_vec1()?;
    assert_eq!(back16, data);
    // i64 → f32 should preserve small values
    let small = Tensor::from_vec(vec![12345i64, -999], 2, dev())?;
    let small_f32: Vec<f32> = small.to_dtype(DType::F32)?.to_vec1()?;
    assert_eq!(small_f32, vec![12345.0, -999.0]);
    Ok(())
}

#[test]
fn where_cond_u32_condition() -> Result<()> {
    // Regression: where_cond must accept non-U8 conditions.
    let cond = Tensor::from_vec(vec![0u32, 1, 0, 2], (2, 2), dev())?;
    let on_true = Tensor::from_vec(vec![10f32, 20.0, 30.0, 40.0], (2, 2), dev())?;
    let on_false = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), dev())?;
    let result = cond.where_cond(&on_true, &on_false)?;
    assert_eq!(result.to_vec2::<f32>()?, vec![vec![1.0, 20.0], vec![3.0, 40.0]]);
    Ok(())
}

// ════════════════════════════════════════════════════════════════════
// Attention (varlen) — vs PyTorch SDPA
// ════════════════════════════════════════════════════════════════════

/// Helper: build varlen inputs and run attention through the Ops trait.
/// Uses file-based I/O via `require_pytorch_ref!` for data exchange.
fn run_varlen_attention_vs_pytorch(
    seq_lens_q: &[usize], seq_lens_k: &[usize],
    num_heads_q: usize, num_heads_k: usize, head_dim: usize,
    causal: bool, py_dtype: &str, dtype: DType, atol: f32, context: &str,
) -> Result<()> {
    // Candle CPU matmul is F32-only; varlen_attention tests all use BF16.
    if !dev().is_cuda() && dtype != DType::F32 {
        eprintln!("SKIPPED ({context}): {dtype:?} matmul is GPU-only on candle");
        return Ok(());
    }
    let total_q: usize = seq_lens_q.iter().sum();
    let total_k: usize = seq_lens_k.iter().sum();
    let batch = seq_lens_q.len();

    let q_data = common::pseudo_random(total_q * num_heads_q * head_dim, 1.0);
    let k_data = common::pseudo_random(total_k * num_heads_k * head_dim, 2.0);
    let v_data = common::pseudo_random(total_k * num_heads_k * head_dim, 3.0);

    // Build cu_seqlens
    let mut cu_q = vec![0u32; batch + 1];
    let mut cu_k = vec![0u32; batch + 1];
    for i in 0..batch {
        cu_q[i + 1] = cu_q[i] + seq_lens_q[i] as u32;
        cu_k[i + 1] = cu_k[i] + seq_lens_k[i] as u32;
    }

    let max_seqlen_q = *seq_lens_q.iter().max().unwrap();
    let max_seqlen_k = *seq_lens_k.iter().max().unwrap();

    let reference = require_pytorch_ref!(
        &[("q", &q_data), ("k", &k_data), ("v", &v_data)],
        &format!(r#"
seq_lens_q = {seq_lens_q:?}
seq_lens_k = {seq_lens_k:?}
nq, nk, hd = {num_heads_q}, {num_heads_k}, {head_dim}
total_q, total_k = sum(seq_lens_q), sum(seq_lens_k)
q_flat = read_input("q", {py_dtype})
k_flat = read_input("k", {py_dtype})
v_flat = read_input("v", {py_dtype})
causal = {causal}
cu_q, cu_k = [0], [0]
for s in seq_lens_q: cu_q.append(cu_q[-1]+s)
for s in seq_lens_k: cu_k.append(cu_k[-1]+s)
outs = []
for b in range(len(seq_lens_q)):
    sq, sk = seq_lens_q[b], seq_lens_k[b]
    qi = q_flat[cu_q[b]*nq*hd:cu_q[b+1]*nq*hd].reshape(sq,nq,hd).transpose(0,1).unsqueeze(0)
    ki = k_flat[cu_k[b]*nk*hd:cu_k[b+1]*nk*hd].reshape(sk,nk,hd).transpose(0,1).unsqueeze(0)
    vi = v_flat[cu_k[b]*nk*hd:cu_k[b+1]*nk*hd].reshape(sk,nk,hd).transpose(0,1).unsqueeze(0)
    if nq != nk:
        ki = ki.repeat_interleave(nq//nk, dim=1)
        vi = vi.repeat_interleave(nq//nk, dim=1)
    if causal:
        co = sk - sq
        amask = torch.zeros(1,1,sq,sk)
        for r in range(sq):
            for c in range(sk):
                if c > co + r: amask[0,0,r,c] = float('-inf')
        oi = torch.nn.functional.scaled_dot_product_attention(qi.float(),ki.float(),vi.float(),attn_mask=amask)
    else:
        oi = torch.nn.functional.scaled_dot_product_attention(qi.float(),ki.float(),vi.float())
    oi = oi.squeeze(0).transpose(0,1).reshape(sq,nq,hd).to({py_dtype})
    outs.append(oi)
out = torch.cat(outs,dim=0).float()
write_output(out)
"#,
            causal = if causal { "True" } else { "False" },
        )
    );

    // Our implementation
    let d = dev();
    let q = Tensor::from_vec(q_data, (total_q, num_heads_q, head_dim), d)?.to_dtype(dtype)?;
    let k = Tensor::from_vec(k_data, (total_k, num_heads_k, head_dim), d)?.to_dtype(dtype)?;
    let v = Tensor::from_vec(v_data, (total_k, num_heads_k, head_dim), d)?.to_dtype(dtype)?;
    let cu_seqlens_q = Tensor::from_vec(cu_q, (batch + 1,), d)?;
    let cu_seqlens_k = Tensor::from_vec(cu_k, (batch + 1,), d)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask = if causal { MaskType::Causal } else { MaskType::Bidirectional };
    let params = VarlenParams {
        cu_seqlens_q: &cu_seqlens_q, cu_seqlens_k: &cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, scale, mask, softcap: None,
    };

    let ops = ops::ops_for(d);
    let out = ops.varlen_attention(&q, &k, &v, &params)?;
    let ours: Vec<f32> = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    common::assert_close(&ours, &reference, atol, context);
    Ok(())
}

#[test]
fn varlen_attention_causal_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[32], &[32], 8, 8, 64, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen causal bf16",
    )
}

#[test]
fn varlen_attention_bidirectional_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[16], &[16], 8, 8, 64, false,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen bidirectional bf16",
    )
}

#[test]
fn varlen_attention_gqa4_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[32], &[32], 32, 8, 128, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen gqa4 bf16",
    )
}

#[test]
fn varlen_attention_gqa8_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[32], &[32], 64, 8, 128, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen gqa8 bf16",
    )
}

#[test]
fn varlen_attention_multi_seq_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[16, 32, 8], &[16, 32, 8], 8, 8, 128, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen multi-seq bf16",
    )
}

#[test]
fn varlen_attention_prefill_decode_vs_pytorch() -> Result<()> {
    // Mixed: seq0 is prefill (32 tokens), seq1 is decode (1 token with 64 KV)
    run_varlen_attention_vs_pytorch(
        &[32, 1], &[32, 64], 8, 8, 128, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen prefill+decode bf16",
    )
}

#[test]
fn varlen_attention_hdim256_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[16], &[16], 8, 8, 256, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen hdim256 bf16",
    )
}

#[test]
fn varlen_attention_gqa32_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[32], &[32], 256, 8, 128, true,
        "torch.bfloat16", DType::BF16, 2e-2, "varlen gqa32 bf16",
    )
}

#[test]
fn varlen_attention_long_seq_vs_pytorch() -> Result<()> {
    run_varlen_attention_vs_pytorch(
        &[256], &[256], 8, 8, 128, true,
        "torch.bfloat16", DType::BF16, 5e-2, "varlen long-seq bf16",
    )
}

#[test]
fn varlen_attention_hdim512_vs_pytorch() -> Result<()> {
    // Gemma4 full_attention layers use head_dim=512 with GQA 8:1
    run_varlen_attention_vs_pytorch(
        &[16], &[16], 8, 1, 512, true,
        "torch.bfloat16", DType::BF16, 5e-2, "varlen hdim512 gqa8 bf16",
    )
}

#[test]
fn varlen_attention_hdim512_short_vs_pytorch() -> Result<()> {
    // Short seq (7 tokens) — tests CUTLASS small-matrix fallback
    run_varlen_attention_vs_pytorch(
        &[7], &[7], 8, 1, 512, true,
        "torch.bfloat16", DType::BF16, 5e-2, "varlen hdim512 short bf16",
    )
}

