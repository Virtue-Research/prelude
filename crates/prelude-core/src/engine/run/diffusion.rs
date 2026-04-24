//! Image/video diffusion scheduling loop.
//!
//! Denoising loop for diffusion models (e.g., Flux, HunyuanVideo).
//! Each step: scheduler provides timestep + noise schedule → Executor runs
//! forward (DiT block) → scheduler updates latents → repeat until denoised.
//!
//! Not yet implemented — placeholder for the paradigm.
