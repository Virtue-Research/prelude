// Standalone kernel microbench for candle-fa3-0102 (pristine FA3 0.10.2, built
// with CUDA 12.9). Dense prefill attention, Qwen3-8B-style shape:
//   batch=128, seqlen in {128,1024,2048,8192}, GQA 32 q / 8 kv heads,
//   head_dim=128, causal, bf16.
// Timing matches microbench_vllm_attn.py: warmup 30, 200 iters, median + p5/p95,
// device-synchronized around each forward. Prints `JSON {...}` for aggregation.
use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_fa3_0102::flash_attn;

// cudarc allocates via cuMemAllocAsync (stream-ordered pool), but the default pool release
// threshold is 0 → every free returns memory to the OS, so every forward re-acquires its
// output/scratch from the driver (the ~220us host overhead seen at small seqlen). Raise the
// release threshold to MAX so the pool retains freed blocks for reuse — the same trick
// PyTorch/vLLM's caching allocator uses. Fair apples-to-apples for kernel-latency comparison.
unsafe extern "C" {
    fn cudaDeviceGetDefaultMemPool(pool: *mut *mut core::ffi::c_void, device: i32) -> i32;
    fn cudaMemPoolSetAttribute(pool: *mut core::ffi::c_void, attr: i32, value: *mut core::ffi::c_void) -> i32;
}
fn retain_mempool() {
    // cudaMemPoolAttrReleaseThreshold = 4; value is a cuuint64_t.
    unsafe {
        let mut pool: *mut core::ffi::c_void = core::ptr::null_mut();
        if cudaDeviceGetDefaultMemPool(&mut pool, 0) == 0 && !pool.is_null() {
            let mut thresh: u64 = u64::MAX;
            let rc = cudaMemPoolSetAttribute(pool, 4, &mut thresh as *mut u64 as *mut core::ffi::c_void);
            eprintln!("[mempool] release-threshold=MAX rc={rc}");
        }
    }
}

const BATCH: usize = 128;
const HQ: usize = 32;
const HKV: usize = 8;
const D: usize = 128;
const SEQLENS: &[usize] = &[128, 1024, 2048, 8192];
const WARMUP: usize = 30;
const ITERS: usize = 200;

fn env_bool(k: &str, default: bool) -> bool {
    std::env::var(k).map(|v| v == "1" || v == "true").unwrap_or(default)
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn seqlens() -> Vec<usize> {
    match std::env::var("SEQLENS") {
        Ok(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => SEQLENS.to_vec(),
    }
}

fn pct(sorted_us: &[f64], p: f64) -> f64 {
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx]
}

fn bench_one(dev: &Device, s: usize, batch: usize, causal: bool, gqa_pack: bool) -> Result<(f64, f64, f64)> {
    let scale = 1.0f32 / (D as f32).sqrt();
    // bf16 zeros via raw alloc (NO to_dtype, NO RNG-cast). candle-core's to_dtype cast kernel
    // overflows its launch config at >=2^32 elements (candle framework bug, unrelated to FA3);
    // candle's bf16 randn internally does f32-randn + to_dtype, so it hits the same bug at S=8192.
    // FA3 forward timing is data-independent (no value-dependent control flow), so zeros inputs
    // give a representative kernel latency and let all 4 seqlens use one consistent method.
    // Input source (FA3 IS mildly data-dependent via online-softmax rescaling, so random
    // != zeros; fair comparison vs vLLM uses random):
    //  LOADIN_DIR=<dir>: load random bf16 q/k/v from <dir>/qkv_s{S}.safetensors (generated
    //    by gen_random_inputs.py via torch — the only way to get random bf16 at S=8192,
    //    where candle's bf16 randn would hit the to_dtype 2^32 cast bug). Pure memcpy load.
    //  RANDIN=1: candle bf16 randn (valid only for total elems < 2^32).
    //  default: bf16 zeros (understates latency ~13%; use only when random is unavailable).
    let (q, k, v) = if let Ok(dir) = std::env::var("LOADIN_DIR") {
        let t = candle::safetensors::load(format!("{dir}/qkv_s{s}.safetensors"), dev)?;
        (t["q"].clone(), t["k"].clone(), t["v"].clone())
    } else if env_bool("RANDIN", false) {
        let (m, sd) = (half::bf16::ZERO, half::bf16::ONE);
        (Tensor::randn(m, sd, (batch, s, HQ, D), dev)?,
         Tensor::randn(m, sd, (batch, s, HKV, D), dev)?,
         Tensor::randn(m, sd, (batch, s, HKV, D), dev)?)
    } else {
        (Tensor::zeros((batch, s, HQ, D), DType::BF16, dev)?,
         Tensor::zeros((batch, s, HKV, D), DType::BF16, dev)?,
         Tensor::zeros((batch, s, HKV, D), DType::BF16, dev)?)
    };

    let run = || -> Result<Tensor> { Ok(flash_attn(&q, &k, &v, scale, causal, gqa_pack)?) };

    let warmup = env_usize("WARMUP", WARMUP);
    let iters = env_usize("ITERS", ITERS);
    for _ in 0..warmup {
        let _ = run()?;
    }
    dev.synchronize()?;

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let o = run()?;
        dev.synchronize()?;
        times.push(t0.elapsed().as_secs_f64() * 1e6); // us
        drop(o);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((pct(&times, 0.50), pct(&times, 0.05), pct(&times, 0.95)))
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    if !env_bool("NO_MEMPOOL", false) {
        retain_mempool();
    }
    let causal = env_bool("CAUSAL", true);
    let gqa_pack = env_bool("GQA_PACK", true);

    // DATASET=<path>: real ragged varlen prefill workload. Each line of the file is one
    // batch = space-separated seqlens (packed <= max_num_batched_tokens by the producer).
    // Runs flash_attn_varlen once per batch; times one full pass over all batches (the total
    // prefill-attention latency for the dataset). Heads default to the topicguard model
    // (Qwen3 GQA 32 q / 4 kv, d128). Inputs: one big random bf16 buffer, sliced per batch.
    if let Ok(path) = std::env::var("DATASET") {
        let v3 = env_bool("V3", false);
        let hq = env_usize("HQ", 32);
        let hkv = env_usize("HKV", 4);
        let scale = 1.0f32 / (D as f32).sqrt();
        let batches: Vec<Vec<usize>> = std::fs::read_to_string(&path)?
            .lines()
            .map(|l| l.split_whitespace().filter_map(|x| x.parse().ok()).collect())
            .filter(|b: &Vec<usize>| !b.is_empty())
            .collect();
        let max_tok = batches.iter().map(|b| b.iter().sum::<usize>()).max().unwrap_or(0);
        // one big random bf16 buffer reused via narrow() (no per-call alloc/gen).
        // DATASET_QKV=<safetensors>: load shared q/k/v (identical to vLLM) instead of randn.
        let (m, sd) = (half::bf16::ZERO, half::bf16::ONE);
        let (q_big, k_big, v_big) = if let Ok(p) = std::env::var("DATASET_QKV") {
            let t = candle::safetensors::load(&p, &dev)?;
            (t["q"].clone(), t["k"].clone(), t["v"].clone())
        } else {
            (Tensor::randn(m, sd, (max_tok, hq, D), &dev)?,
             Tensor::randn(m, sd, (max_tok, hkv, D), &dev)?,
             Tensor::randn(m, sd, (max_tok, hkv, D), &dev)?)
        };
        // pre-build cu_seqlens (u32) per batch
        let cus: Vec<(Tensor, usize)> = batches.iter().map(|b| {
            let mut cu = Vec::with_capacity(b.len() + 1);
            let mut acc = 0u32;
            cu.push(0u32);
            for &l in b { acc += l as u32; cu.push(acc); }
            let max_s = *b.iter().max().unwrap();
            (Tensor::new(cu.as_slice(), &dev).unwrap(), max_s)
        }).collect();
        let total_tok: usize = batches.iter().map(|b| b.iter().sum::<usize>()).sum();
        println!("DATASET {path}: {} batches, {} total tokens, GQA {hq}/{hkv}, d={D}, causal={causal}, bf16",
            batches.len(), total_tok);

        let one_pass = || -> Result<()> {
            for (i, b) in batches.iter().enumerate() {
                let total: usize = b.iter().sum();
                let (cu, max_s) = &cus[i];
                let q = q_big.narrow(0, 0, total)?;
                let k = k_big.narrow(0, 0, total)?;
                let v = v_big.narrow(0, 0, total)?;
                let _o = if v3 {
                    candle_fa3_0102::flash_attn_varlen_v3(&q, &k, &v, cu, cu, *max_s, *max_s, scale, causal)?
                } else {
                    candle_fa3_0102::flash_attn_varlen(&q, &k, &v, cu, cu, *max_s, *max_s, scale, causal, gqa_pack)?
                };
            }
            Ok(())
        };
        let warmup = env_usize("WARMUP", 2);
        let iters = env_usize("ITERS", 10);
        for _ in 0..warmup { one_pass()?; }
        dev.synchronize()?;
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            one_pass()?;
            dev.synchronize()?;
            times.push(t0.elapsed().as_secs_f64() * 1e3); // ms per full pass
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = pct(&times, 0.50);
        println!("candle-fa3-0102 DATASET: {med:.2} ms/pass (median of {iters}) [p5 {:.2}, p95 {:.2}] | {:.1} us/batch",
            pct(&times, 0.05), pct(&times, 0.95), med * 1000.0 / batches.len() as f64);
        println!("JSON {{\"ms_per_pass\": {med:.2}, \"batches\": {}, \"total_tokens\": {total_tok}}}", batches.len());
        return Ok(());
    }

    // XCHECK: load shared q/k/v (s,h,d) bf16 from safetensors, run candle FA3, save output.
    // Used by xcheck_vs_vllm.py to compare candle vs vLLM-FA3 vs an fp32 eager reference.
    if env_bool("XCHECK", false) {
        let inp = std::env::var("XCHECK_IN").unwrap_or_else(|_| "/tmp/fa_qkv.safetensors".into());
        let outp = std::env::var("XCHECK_OUT").unwrap_or_else(|_| "/tmp/fa_candle_out.safetensors".into());
        let t = candle::safetensors::load(&inp, &dev)?;
        let scale = 1.0f32 / (D as f32).sqrt();
        let o = if env_bool("V3", false) {
            // newer vllm-fa hopper varlen path (single-seq here)
            let s = t["q"].dims()[0] as u32;
            let cu = Tensor::new(&[0u32, s], &dev)?;
            candle_fa3_0102::flash_attn_varlen_v3(&t["q"], &t["k"], &t["v"], &cu, &cu, s as usize, s as usize, scale, causal)?
        } else if env_bool("XVARLEN", false) {
            // single-sequence varlen: q/k/v are (s,h,d) flat, cu_seqlens=[0,s]
            let s = t["q"].dims()[0] as u32;
            let cu = Tensor::new(&[0u32, s], &dev)?;
            candle_fa3_0102::flash_attn_varlen(&t["q"], &t["k"], &t["v"], &cu, &cu, s as usize, s as usize, scale, causal, gqa_pack)?
        } else {
            let q = t["q"].unsqueeze(0)?; // (s,h,d) -> (1,s,h,d)
            let k = t["k"].unsqueeze(0)?;
            let v = t["v"].unsqueeze(0)?;
            flash_attn(&q, &k, &v, scale, causal, gqa_pack)?.squeeze(0)?
        };
        dev.synchronize()?;
        let mut m = std::collections::HashMap::new();
        m.insert("out".to_string(), o.to_dtype(DType::F32)?);
        candle::safetensors::save(&m, &outp)?;
        println!("XCHECK candle out saved: {outp} dims={:?} causal={causal} gqa_pack={gqa_pack}", t["q"].dims());
        return Ok(());
    }

    // SMOKE=1: one tiny forward to isolate which (causal,gqa_pack) faults.
    // VARLEN=1: use the varlen path (flat total_tokens + cu_seqlens) instead of dense.
    if env_bool("SMOKE", false) {
        let s: usize = std::env::var("S").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let b: usize = std::env::var("B").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
        let scale = 1.0f32 / (D as f32).sqrt();
        if env_bool("VARLEN", false) {
            let total = b * s;
            println!("SMOKE VARLEN b={b} s={s} total={total} causal={causal} gqa_pack={gqa_pack} ...");
            let q = Tensor::randn(0f32, 1f32, (total, HQ, D), &dev)?.to_dtype(DType::BF16)?;
            let k = Tensor::randn(0f32, 1f32, (total, HKV, D), &dev)?.to_dtype(DType::BF16)?;
            let v = Tensor::randn(0f32, 1f32, (total, HKV, D), &dev)?.to_dtype(DType::BF16)?;
            let cu: Vec<u32> = (0..=b).map(|i| (i * s) as u32).collect();
            let cu = Tensor::new(cu.as_slice(), &dev)?;
            let o = candle_fa3_0102::flash_attn_varlen(&q, &k, &v, &cu, &cu, s, s, scale, causal, gqa_pack)?;
            dev.synchronize()?;
            println!("SMOKE OK varlen out={:?}", o.dims());
            return Ok(());
        }
        let qn = b * s * HQ * D;
        // ZINPUT=1: build bf16 inputs via raw zeros alloc (NO to_dtype cast), to test
        // whether the FA3 kernel itself handles >=2^32-element tensors (candle's to_dtype
        // cast kernel overflows at 2^32; this isolates FA3 from that framework bug).
        if env_bool("ZINPUT", false) {
            println!("SMOKE ZINPUT b={b} s={s} (q_elems={qn}) causal={causal} gqa_pack={gqa_pack} ...");
            let q = Tensor::zeros((b, s, HQ, D), DType::BF16, &dev)?;
            let k = Tensor::zeros((b, s, HKV, D), DType::BF16, &dev)?;
            let v = Tensor::zeros((b, s, HKV, D), DType::BF16, &dev)?;
            dev.synchronize()?;
            eprintln!("[diag] bf16 zeros inputs OK (no cast)");
            match flash_attn(&q, &k, &v, scale, causal, gqa_pack) {
                Err(e) => eprintln!("[diag] FAIL inside flash_attn: {e}"),
                Ok(o) => match dev.synchronize() {
                    Err(e) => eprintln!("[diag] FAIL at synchronize (=> FA3 KERNEL LAUNCH): {e}"),
                    Ok(_) => println!("SMOKE OK: FA3 kernel ran at {qn} elems, out={:?}", o.dims()),
                },
            }
            return Ok(());
        }
        println!("SMOKE b={b} s={s} causal={causal} gqa_pack={gqa_pack} (q_elems={qn}) ...");
        // Step 1: zeros (raw alloc, no fill kernel, no RNG)
        match Tensor::zeros((b, s, HQ, D), DType::F32, &dev).and_then(|t| { dev.synchronize()?; Ok(t) }) {
            Err(e) => { eprintln!("[diag] FAIL at zeros f32 (raw alloc): {e}"); return Ok(()); }
            Ok(_) => eprintln!("[diag] zeros f32 OK (alloc of {qn} elems fine)"),
        }
        // Step 2: randn (RNG fill kernel over qn elems)
        let qf = match Tensor::randn(0f32, 1f32, (b, s, HQ, D), &dev).and_then(|t| { dev.synchronize()?; Ok(t) }) {
            Err(e) => { eprintln!("[diag] FAIL at randn f32 (RNG fill kernel): {e}"); return Ok(()); }
            Ok(t) => { eprintln!("[diag] randn f32 OK"); t }
        };
        // Step 3: to_dtype bf16 (cast kernel over qn elems)
        let q = match qf.to_dtype(DType::BF16).and_then(|t| { dev.synchronize()?; Ok(t) }) {
            Err(e) => { eprintln!("[diag] FAIL at to_dtype bf16 (cast kernel): {e}"); return Ok(()); }
            Ok(t) => { eprintln!("[diag] to_dtype bf16 OK"); t }
        };
        let k = Tensor::randn(0f32, 1f32, (b, s, HKV, D), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (b, s, HKV, D), &dev)?.to_dtype(DType::BF16)?;
        dev.synchronize()?;
        eprintln!("[diag] inputs allocated OK");
        match flash_attn(&q, &k, &v, scale, causal, gqa_pack) {
            Err(e) => { eprintln!("[diag] FAIL INSIDE flash_attn (candle alloc/op, before kernel): {e}"); return Ok(()); }
            Ok(o) => {
                eprintln!("[diag] flash_attn returned Ok (run_mha is void; launch error would be sticky)");
                match dev.synchronize() {
                    Err(e) => { eprintln!("[diag] FAIL at post-call synchronize (=> KERNEL LAUNCH sticky error): {e}"); return Ok(()); }
                    Ok(_) => { println!("SMOKE OK out={:?}", o.dims()); return Ok(()); }
                }
            }
        }
    }

    println!(
        "config: dense batch={BATCH}, GQA {HQ}/{HKV}, d={D}, causal={causal}, bf16, \
         gqa_packing={gqa_pack} | warmup {WARMUP} iters {ITERS}"
    );
    let mut json = String::from("{");
    let sl = seqlens();
    for (i, &s) in sl.iter().enumerate() {
        let (med, p5, p95) = bench_one(&dev, s, BATCH, causal, gqa_pack)?;
        println!("candle-fa3-0102 seqlen={s:5}: {med:9.1} us/fwd   [p5 {p5:.1}, p95 {p95:.1}]");
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("\"{s}\": {med:.1}"));
    }
    json.push('}');
    println!("JSON {json}");
    Ok(())
}
