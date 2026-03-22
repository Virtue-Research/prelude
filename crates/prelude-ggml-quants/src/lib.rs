//! llama.cpp FFI backend for GGUF inference.
//!
//! Thin Rust wrapper around llama.cpp's C API via a C shim (`llama_ffi.c`).
//! Statically links libllama.a + libggml*.a (~10MB).

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::Once;

pub type llama_token = i32;

const DEFAULT_PHYSICAL_CORES: i32 = 8;

/// Estimate physical core count (half of logical CPUs on hyperthreaded systems).
fn physical_core_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1) as i32)
        .unwrap_or(DEFAULT_PHYSICAL_CORES)
}

// ── Opaque C types ──────────────────────────────────────────────────────

#[repr(C)]
pub struct llama_model { _opaque: [u8; 0] }
#[repr(C)]
pub struct llama_context { _opaque: [u8; 0] }
#[repr(C)]
pub struct llama_vocab { _opaque: [u8; 0] }

// ── FFI declarations ────────────────────────────────────────────────────

unsafe extern "C" {
    // Backend init/free
    fn llama_backend_init();
    fn llama_ffi_set_omp_threads(n: i32);

    // Our C wrapper functions (llama_ffi.c)
    fn llama_ffi_load_model(path: *const i8, n_gpu_layers: i32) -> *mut llama_model;
    fn llama_ffi_create_context(
        model: *mut llama_model, n_ctx: i32, n_batch: i32, n_threads: i32,
    ) -> *mut llama_context;
    fn llama_ffi_decode(
        ctx: *mut llama_context, tokens: *const i32, n_tokens: i32,
        logits_out: *mut f32, n_vocab: i32,
    ) -> i32;
    fn llama_ffi_clear_cache(ctx: *mut llama_context);
    fn llama_ffi_profile(
        ctx: *mut llama_context, vocab: *const llama_vocab,
        prompt_tokens: *const i32, n_prompt: i32,
        max_new: i32, n_vocab: i32,
    );
    fn llama_ffi_generate(
        ctx: *mut llama_context, vocab: *const llama_vocab,
        prompt_tokens: *const i32, n_prompt: i32,
        out_tokens: *mut i32, out_logits: *mut f32,
        max_new: i32, n_vocab: i32,
    ) -> i32;

    // Direct llama.h functions we still need
    fn llama_model_free(model: *mut llama_model);
    fn llama_free(ctx: *mut llama_context);
    fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;
    fn llama_tokenize(
        vocab: *const llama_vocab, text: *const i8, text_len: i32,
        tokens: *mut llama_token, n_tokens_max: i32,
        add_special: bool, parse_special: bool,
    ) -> i32;
    fn llama_token_to_piece(
        vocab: *const llama_vocab, token: llama_token,
        buf: *mut i8, length: i32, lstrip: i32, special: bool,
    ) -> i32;
    fn llama_model_n_ctx_train(model: *const llama_model) -> i32;
    fn llama_model_n_embd(model: *const llama_model) -> i32;
    fn llama_model_n_layer(model: *const llama_model) -> i32;
    fn llama_model_n_head(model: *const llama_model) -> i32;
    fn llama_model_n_head_kv(model: *const llama_model) -> i32;
    fn llama_model_chat_template(model: *const llama_model, name: *const i8) -> *const i8;
    fn llama_n_ctx(ctx: *const llama_context) -> u32;
}

// ── Safe wrappers ───────────────────────────────────────────────────────

static INIT: Once = Once::new();

pub fn backend_init() {
    INIT.call_once(|| {
        // Limit OpenMP thread pool to physical cores before any ggml work.
        // Without this, OpenMP defaults to all logical CPUs (including hyperthreads),
        // causing severe contention with tokio/rayon threads (~10x decode slowdown).
        if std::env::var("OMP_NUM_THREADS").is_err() {
            let physical_cores = physical_core_count();
            unsafe { llama_ffi_set_omp_threads(physical_cores) };
        }
        unsafe { llama_backend_init() };
    });
}

pub struct LlamaModel {
    ptr: *mut llama_model,
}

unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() { unsafe { llama_model_free(self.ptr) }; }
    }
}

impl LlamaModel {
    pub fn load(path: &Path, n_gpu_layers: i32) -> Result<Self, String> {
        backend_init();
        let c = CString::new(path.to_str().ok_or("invalid path")?).map_err(|e| e.to_string())?;
        let ptr = unsafe { llama_ffi_load_model(c.as_ptr(), n_gpu_layers) };
        if ptr.is_null() { return Err(format!("failed to load: {}", path.display())); }
        Ok(Self { ptr })
    }

    pub fn vocab(&self) -> *const llama_vocab { unsafe { llama_model_get_vocab(self.ptr) } }
    pub fn n_vocab(&self) -> usize { unsafe { llama_vocab_n_tokens(self.vocab()) as usize } }
    pub fn eos_token(&self) -> llama_token { unsafe { llama_vocab_eos(self.vocab()) } }
    pub fn bos_token(&self) -> llama_token { unsafe { llama_vocab_bos(self.vocab()) } }
    pub fn is_eog(&self, token: llama_token) -> bool { unsafe { llama_vocab_is_eog(self.vocab(), token) } }
    pub fn n_embd(&self) -> usize { unsafe { llama_model_n_embd(self.ptr) as usize } }
    pub fn n_layer(&self) -> usize { unsafe { llama_model_n_layer(self.ptr) as usize } }
    pub fn n_ctx_train(&self) -> usize { unsafe { llama_model_n_ctx_train(self.ptr) as usize } }
    pub fn n_head(&self) -> usize { unsafe { llama_model_n_head(self.ptr) as usize } }
    pub fn n_head_kv(&self) -> usize { unsafe { llama_model_n_head_kv(self.ptr) as usize } }

    pub fn chat_template(&self) -> Option<String> {
        let p = unsafe { llama_model_chat_template(self.ptr, ptr::null()) };
        if p.is_null() { None } else { Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()) }
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<llama_token> {
        let c = CString::new(text).unwrap_or_default();
        let mut buf = vec![0i32; text.len() + 32];
        let n = unsafe { llama_tokenize(self.vocab(), c.as_ptr(), text.len() as i32, buf.as_mut_ptr(), buf.len() as i32, add_special, false) };
        if n < 0 {
            buf.resize((-n) as usize, 0);
            let n = unsafe { llama_tokenize(self.vocab(), c.as_ptr(), text.len() as i32, buf.as_mut_ptr(), buf.len() as i32, add_special, false) };
            buf.truncate(n.max(0) as usize);
        } else {
            buf.truncate(n as usize);
        }
        buf
    }

    pub fn token_to_str(&self, token: llama_token) -> String {
        let mut buf = vec![0u8; 128];
        let n = unsafe { llama_token_to_piece(self.vocab(), token, buf.as_mut_ptr() as *mut _, buf.len() as i32, 0, true) };
        if n > 0 { buf.truncate(n as usize); String::from_utf8_lossy(&buf).into_owned() } else { String::new() }
    }

    pub fn as_ptr(&self) -> *mut llama_model { self.ptr }
}

pub struct LlamaContext {
    ptr: *mut llama_context,
    n_vocab: usize,
}

unsafe impl Send for LlamaContext {}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() { unsafe { llama_free(self.ptr) }; }
    }
}

impl LlamaContext {
    pub fn new(model: &LlamaModel, n_ctx: u32, n_batch: u32) -> Result<Self, String> {
        // Use physical core count (half of available_parallelism on hyperthreaded systems)
        let n_threads = physical_core_count();
        let ptr = unsafe { llama_ffi_create_context(model.as_ptr(), n_ctx as i32, n_batch as i32, n_threads) };
        if ptr.is_null() { return Err("failed to create context".into()); }
        Ok(Self { ptr, n_vocab: model.n_vocab() })
    }

    pub fn decode_tokens(&mut self, tokens: &[llama_token]) -> Result<Vec<f32>, String> {
        let mut logits = vec![0.0f32; self.n_vocab];
        let rc = unsafe { llama_ffi_decode(self.ptr, tokens.as_ptr(), tokens.len() as i32, logits.as_mut_ptr(), self.n_vocab as i32) };
        if rc != 0 { return Err(format!("llama_decode failed: {rc}")); }
        Ok(logits)
    }

    /// Generate tokens entirely in C (prefill + decode loop).
    /// Returns (generated_token_ids, last_logits).
    pub fn generate(
        &mut self,
        vocab: *const llama_vocab,
        prompt_tokens: &[llama_token],
        max_new: usize,
    ) -> Result<(Vec<llama_token>, Vec<f32>), String> {
        let mut out_tokens = vec![0i32; max_new];
        let mut out_logits = vec![0.0f32; self.n_vocab];
        let n = unsafe {
            llama_ffi_generate(
                self.ptr, vocab,
                prompt_tokens.as_ptr(), prompt_tokens.len() as i32,
                out_tokens.as_mut_ptr(), out_logits.as_mut_ptr(),
                max_new as i32, self.n_vocab as i32,
            )
        };
        if n < 0 { return Err(format!("llama_ffi_generate failed: {n}")); }
        out_tokens.truncate(n as usize);
        Ok((out_tokens, out_logits))
    }

    pub fn clear_kv_cache(&mut self) {
        unsafe { llama_ffi_clear_cache(self.ptr) };
    }

    pub fn n_ctx(&self) -> u32 { unsafe { llama_n_ctx(self.ptr) } }

    pub fn profile(&mut self, vocab: *const llama_vocab, prompt_tokens: &[llama_token], max_new: usize) {
        unsafe {
            llama_ffi_profile(
                self.ptr, vocab,
                prompt_tokens.as_ptr(), prompt_tokens.len() as i32,
                max_new as i32, self.n_vocab as i32,
            );
        }
    }
}
