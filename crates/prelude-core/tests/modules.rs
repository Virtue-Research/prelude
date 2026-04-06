//! Integration tests for modules and fused composite ops.
//!
//! Precision tests compare against PyTorch (via subprocess) as ground truth.
//! Tests requiring PyTorch are skipped if python3/torch is not available.

mod common;

use prelude_core::tensor::{DType, Device, Module, Result, Tensor};

// == Linear ==

#[test]
fn linear_no_bias_vs_pytorch() -> Result<()> {
    let w_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let x_data = vec![1f32; 8];

    let ref_flat = require_pytorch_ref!(
        &[("w", &w_data), ("x", &x_data)],
        r#"
w = read_input("w").reshape(3, 4)
x = read_input("x").reshape(2, 4)
y = x @ w.T
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 3);

    let w = Tensor::from_vec(w_data, (3, 4), &Device::Cpu)?;
    let x = Tensor::from_vec(x_data, (2, 4), &Device::Cpu)?;
    let linear = prelude_core::models::commons::linear::Linear::from_weight(w, None)?;
    let y = linear.forward(
        &x,
        &prelude_core::models::commons::BatchState::no_lora(),
        prelude_core::ops::select_ops(&Device::Cpu),
    )?;

    let ours = y.to_vec2::<f32>()?;
    common::assert_close_2d(&ours, &reference, 1e-5, "linear_no_bias");
    Ok(())
}

#[test]
fn linear_with_bias_vs_pytorch() -> Result<()> {
    let ref_flat = require_pytorch_ref!(
        &[],
        r#"
w = torch.ones(2, 3)
b = torch.tensor([10.0, 20.0])
x = torch.tensor([[1.0, 2.0, 3.0]])
y = x @ w.T + b
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 2);

    let w = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    let b = Tensor::from_vec(vec![10f32, 20.], 2, &Device::Cpu)?;
    let x = Tensor::from_vec(vec![1f32, 2., 3.], (1, 3), &Device::Cpu)?;
    let linear = prelude_core::models::commons::linear::Linear::from_weight(w, Some(b))?;
    let y = linear.forward(
        &x,
        &prelude_core::models::commons::BatchState::no_lora(),
        prelude_core::ops::select_ops(&Device::Cpu),
    )?;

    let ours = y.to_vec2::<f32>()?;
    common::assert_close(&ours[0], &reference[0], 1e-5, "linear_with_bias");
    Ok(())
}

#[test]
fn linear_multi_dtype_vs_pytorch() -> Result<()> {
    let batch = 4;
    let in_dim = 256;
    let out_dim = 128;
    let x_data = common::pseudo_random(batch * in_dim, 25.0);
    let w_data = common::pseudo_random(out_dim * in_dim, 26.0);

    for cfg in common::ALL_DTYPES {
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

        let x = Tensor::from_vec(x_data.clone(), (batch, in_dim), &Device::Cpu)?
            .to_dtype(cfg.dtype)?;
        let w = Tensor::from_vec(w_data.clone(), (out_dim, in_dim), &Device::Cpu)?
            .to_dtype(cfg.dtype)?;
        let linear = prelude_core::models::commons::linear::Linear::from_weight(w, None)?;
        let y = linear
            .forward(
                &x,
                &prelude_core::models::commons::BatchState::no_lora(),
                prelude_core::ops::select_ops(&Device::Cpu),
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

// == RmsNorm ==

#[test]
fn rmsnorm_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 2., 3., 4.];
    let w_data = vec![1f32, 1., 1., 1.];

    let ref_flat = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 4)
w = read_input("w")
y = torch.nn.functional.rms_norm(x, (4,), w, 1e-6)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 4);

    let x = Tensor::from_vec(x_data, (1, 4), &Device::Cpu)?;
    let w = Tensor::from_vec(w_data, (4,), &Device::Cpu)?;
    let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(w, 1e-6);
    let y = norm.forward(&x)?;

    let ours = y.to_vec2::<f32>()?;
    common::assert_close(&ours[0], &reference[0], 1e-5, "rmsnorm_unit_weight");
    Ok(())
}

#[test]
fn rmsnorm_with_weight_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 1., 1., 1.];
    let w_data = vec![2f32, 0.5, 1.0, 3.0];

    let ref_flat = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 4)
w = read_input("w")
y = torch.nn.functional.rms_norm(x, (4,), w, 1e-6)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 4);

    let x = Tensor::from_vec(x_data, (1, 4), &Device::Cpu)?;
    let w = Tensor::from_vec(w_data, (4,), &Device::Cpu)?;
    let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(w, 1e-6);
    let y = norm.forward(&x)?;

    let ours = y.to_vec2::<f32>()?;
    common::assert_close(&ours[0], &reference[0], 1e-5, "rmsnorm_with_weight");
    Ok(())
}

#[test]
fn rmsnorm_multi_dtype_vs_pytorch() -> Result<()> {
    let batch = 4;
    let hidden = 512;
    let x_data = common::pseudo_random(batch * hidden, 23.0);
    let w_data = common::pseudo_random(hidden, 24.0);
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

        let x = Tensor::from_vec(x_data.clone(), (batch, hidden), &Device::Cpu)?
            .to_dtype(cfg.dtype)?;
        let w =
            Tensor::from_vec(w_data.clone(), (hidden,), &Device::Cpu)?.to_dtype(cfg.dtype)?;
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

// == Embedding ==

#[test]
fn embedding_vs_pytorch() -> Result<()> {
    let ref_flat = require_pytorch_ref!(
        &[],
        r#"
table = torch.arange(12, dtype=torch.float32).reshape(4, 3)
ids = torch.tensor([2, 0, 3])
y = torch.nn.functional.embedding(ids, table)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 3);

    let table = Tensor::from_vec(
        (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        (4, 3),
        &Device::Cpu,
    )?;
    let emb = prelude_core::models::commons::embedding::Embedding::new(table, 3);
    let ids = Tensor::from_vec(vec![2u32, 0, 3], 3, &Device::Cpu)?;
    let y = emb.forward(&ids)?;

    let ours = y.to_vec2::<f32>()?;
    common::assert_close_2d(&ours, &reference, 1e-6, "embedding");
    Ok(())
}

// == Fused composite ops ==

#[test]
fn fused_add_rmsnorm_vs_pytorch() -> Result<()> {
    let x_data = vec![1f32, 2., 3., 4.];
    let h_data = vec![0.5f32, 0.5, 0.5, 0.5];
    let w_data = vec![1f32, 1., 1., 1.];

    let result = require_pytorch_ref_multi!(
        &[("x", &x_data), ("h", &h_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 4)
h = read_input("h").reshape(1, 4)
w = read_input("w")
residual = x + h
normed = torch.nn.functional.rms_norm(residual, (4,), w, 1e-6)
write_outputs(residual=residual, normed=normed)
"#
    );
    let ref_residual = &result["residual"];
    let ref_normed = &result["normed"];

    let x = Tensor::from_vec(x_data, (1, 4), &Device::Cpu)?;
    let h = Tensor::from_vec(h_data, (1, 4), &Device::Cpu)?;
    let w = Tensor::from_vec(w_data.clone(), (4,), &Device::Cpu)?;
    let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(w.clone(), 1e-6);
    let ops = prelude_core::ops::select_ops(&Device::Cpu);

    let (residual, normed) =
        ops.add_rmsnorm(&x, &h, &w, 1e-6 as f32)?;

    common::assert_close(
        &residual.to_vec2::<f32>()?[0],
        &ref_residual,
        1e-5,
        "add_rmsnorm residual",
    );
    common::assert_close(
        &normed.to_vec2::<f32>()?[0],
        &ref_normed,
        1e-5,
        "add_rmsnorm normed",
    );
    Ok(())
}

#[test]
fn fused_silu_mul_vs_pytorch() -> Result<()> {
    let gate_data = vec![0f32, 1., -1., 2., -2., 0.5];
    let up_data = vec![1f32, 2., 3., 0.5, 1., 4.];

    let ref_flat = require_pytorch_ref!(
        &[("gate", &gate_data), ("up", &up_data)],
        r#"
gate = read_input("gate").reshape(1, 6)
up = read_input("up").reshape(1, 6)
y = torch.nn.functional.silu(gate) * up
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 6);

    let gate = Tensor::from_vec(gate_data, (1, 6), &Device::Cpu)?;
    let up = Tensor::from_vec(up_data, (1, 6), &Device::Cpu)?;
    let ops = prelude_core::ops::select_ops(&Device::Cpu);

    let y = ops.silu_mul(&gate, &up)?;

    common::assert_close(&y.to_vec2::<f32>()?[0], &reference[0], 1e-5, "silu_mul");
    Ok(())
}

#[test]
fn fused_rms_norm_vs_pytorch() -> Result<()> {
    let x_data = vec![3f32, 4.];
    let w_data = vec![1f32, 1.];

    let ref_flat = require_pytorch_ref!(
        &[("x", &x_data), ("w", &w_data)],
        r#"
x = read_input("x").reshape(1, 2)
w = read_input("w")
y = torch.nn.functional.rms_norm(x, (2,), w, 1e-6)
write_output(y)
"#
    );
    let reference = common::unflatten(&ref_flat, 2);

    let x = Tensor::from_vec(x_data, (1, 2), &Device::Cpu)?;
    let w = Tensor::from_vec(w_data, (2,), &Device::Cpu)?;
    let norm = prelude_core::models::commons::linear::RmsNorm::from_weight(w.clone(), 1e-6);
    let ops = prelude_core::ops::select_ops(&Device::Cpu);

    let y = ops.rms_norm(&x, &w, 1e-6 as f32)?;

    common::assert_close(&y.to_vec2::<f32>()?[0], &reference[0], 1e-5, "rms_norm");
    Ok(())
}
