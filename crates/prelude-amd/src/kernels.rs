//! AMD GPU kernel dispatch via t0-gpu.
//!
//! Each function takes raw GpuBuffers + dimensions and dispatches
//! t0-gpu kernels. Called from AmdOps hook methods.

#[cfg(feature = "rocm")]
use t0_gpu::kfd::GpuBuffer;
#[cfg(feature = "rocm")]
use t0_gpu::ignis::gpu_context::GpuRuntime;
#[cfg(feature = "rocm")]
use t0_gpu::t0::block_dsl::{BlockKernel, BVal};
#[cfg(feature = "rocm")]
use t0_gpu::t0::ir::Target;
#[cfg(feature = "rocm")]
use std::sync::Arc;

// ── Elementwise: add ────────────────────────────────────────────

#[cfg(feature = "rocm")]
pub fn add_f32(rt: &Arc<GpuRuntime>, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_binary_kernel(rt, "bdsl_add_f32", |kb, va, vb| va.add(kb, vb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![a.va_addr => u64, b.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn sub_f32(rt: &Arc<GpuRuntime>, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_binary_kernel(rt, "bdsl_sub_f32", |kb, va, vb| va.sub(kb, vb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![a.va_addr => u64, b.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn mul_f32(rt: &Arc<GpuRuntime>, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_binary_kernel(rt, "bdsl_mul_f32", |kb, va, vb| va.mul(kb, vb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![a.va_addr => u64, b.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn div_f32(rt: &Arc<GpuRuntime>, a: &GpuBuffer, b: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_binary_kernel(rt, "bdsl_div_f32", |kb, va, vb| va.div(kb, vb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![a.va_addr => u64, b.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

// ── Elementwise: unary ──────────────────────────────────────────

#[cfg(feature = "rocm")]
pub fn exp_f32(rt: &Arc<GpuRuntime>, x: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_unary_kernel(rt, "bdsl_exp_f32", |kb, v| v.exp(kb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![x.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn neg_f32(rt: &Arc<GpuRuntime>, x: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_unary_kernel(rt, "bdsl_neg_f32", |kb, v| v.neg(kb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![x.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn sqrt_f32(rt: &Arc<GpuRuntime>, x: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_unary_kernel(rt, "bdsl_sqrt_f32", |kb, v| v.sqrt(kb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![x.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn recip_f32(rt: &Arc<GpuRuntime>, x: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_unary_kernel(rt, "bdsl_rcp_f32", |kb, v| v.rcp(kb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![x.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

#[cfg(feature = "rocm")]
pub fn abs_f32(rt: &Arc<GpuRuntime>, x: &GpuBuffer, n: usize) -> Result<GpuBuffer, String> {
    let out = rt.alloc_f32(n)?;
    let kernel = ensure_unary_kernel(rt, "bdsl_abs_f32", |kb, v| v.abs(kb))?;
    let grid_x = ceildiv(n, 256) as u32 * 256;
    let ka = t0_gpu::kernargs![x.va_addr => u64, out.va_addr => u64, n as u32 => u32];
    rt.dispatch(&kernel, [grid_x, 1, 1], &ka)?;
    Ok(out)
}

// ── GEMM: C = A @ B (BF16 WMMA) ────────────────────────────────

#[cfg(feature = "rocm")]
pub fn gemm_bf16(
    rt: &Arc<GpuRuntime>,
    a: &GpuBuffer, b: &GpuBuffer,
    m: usize, k: usize, n: usize,
) -> Result<GpuBuffer, String> {
    use t0_gpu::t0::gemm_gen;

    let out = rt.alloc_f32(m * n)?;
    let cfg = gemm_gen::auto_select(m as u32, k as u32, n as u32);
    let kernel = rt.ensure_kernel_t0(
        &cfg.name(),
        || gemm_gen::generate(&cfg),
        [cfg.wg_size as u32, 1, 1],
        cfg.lds_total(),
    )?;

    let ka = gemm_gen::build_kernargs(
        a.va_addr, b.va_addr, out.va_addr,
        k as u32, n as u32, m as u32, &cfg,
    );
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m as u32, n as u32);
    rt.dispatch(&kernel, [gx, gy, 1], &ka)?;
    Ok(out)
}

// ── Memory transfer ─────────────────────────────────────────────

#[cfg(feature = "rocm")]
pub fn upload_f32(rt: &Arc<GpuRuntime>, data: &[f32]) -> Result<GpuBuffer, String> {
    rt.upload_f32(data)
}

// ── Helpers ─────────────────────────────────────────────────────

#[cfg(feature = "rocm")]
fn ceildiv(a: usize, b: usize) -> usize { (a + b - 1) / b }

/// Compile & cache a unary elementwise kernel (input, output, n).
#[cfg(feature = "rocm")]
fn ensure_unary_kernel(
    rt: &Arc<GpuRuntime>,
    name: &str,
    body: impl FnOnce(&mut BlockKernel, BVal) -> BVal,
) -> Result<Arc<t0_gpu::kfd::GpuKernel>, String> {
    if let Some(k) = rt.get_kernel(name) { return Ok(k); }
    let mut kb = BlockKernel::new(name, 256);
    let in_ptr = kb.arg_ptr("input");
    let out_ptr = kb.arg_ptr("output");
    let n = kb.arg_u32("n");
    let pid = kb.program_id(0);
    let bs = kb.const_u32(256);
    let base = pid.mul(&mut kb, bs);
    let tid = kb.arange(0, 256);
    let off = tid.add(&mut kb, base);
    let val = kb.load_checked(in_ptr, off, n);
    let result = body(&mut kb, val);
    kb.store_checked(out_ptr, off, result, n);
    let compiled = kb.compile(Target::GFX1100)?;
    rt.compile_dsl(compiled)
}

/// Compile & cache a binary elementwise kernel (a, b, output, n).
#[cfg(feature = "rocm")]
fn ensure_binary_kernel(
    rt: &Arc<GpuRuntime>,
    name: &str,
    body: impl FnOnce(&mut BlockKernel, BVal, BVal) -> BVal,
) -> Result<Arc<t0_gpu::kfd::GpuKernel>, String> {
    if let Some(k) = rt.get_kernel(name) { return Ok(k); }
    let mut kb = BlockKernel::new(name, 256);
    let a_ptr = kb.arg_ptr("a");
    let b_ptr = kb.arg_ptr("b");
    let out_ptr = kb.arg_ptr("out");
    let n = kb.arg_u32("n");
    let pid = kb.program_id(0);
    let bs = kb.const_u32(256);
    let base = pid.mul(&mut kb, bs);
    let tid = kb.arange(0, 256);
    let off = tid.add(&mut kb, base);
    let va = kb.load_checked(a_ptr, off, n);
    let vb = kb.load_checked(b_ptr, off, n);
    let result = body(&mut kb, va, vb);
    kb.store_checked(out_ptr, off, result, n);
    let compiled = kb.compile(Target::GFX1100)?;
    rt.compile_dsl(compiled)
}
