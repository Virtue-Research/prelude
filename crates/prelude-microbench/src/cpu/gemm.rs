use candle_core::{Device, Result, Tensor};
use half::bf16;
use std::time::Instant;

use crate::report::{BenchEntry, BenchReport};

pub fn bench_all(report: &mut BenchReport, warmup: usize, repeats: usize) -> Result<()> {
    let gemm_configs: &[(usize, usize, usize)] = &[
        (1, 896, 4864),
        (4, 896, 4864),
        (16, 896, 4864),
        (64, 896, 4864),
        (128, 896, 4864),
        (256, 896, 4864),
        (1, 1536, 8960),
        (16, 1536, 8960),
        (64, 1536, 8960),
        (128, 1536, 8960),
        (1, 3584, 18944),
        (16, 3584, 18944),
        (64, 3584, 18944),
        (1, 7168, 38656),
        (16, 7168, 38656),
        (64, 7168, 38656),
    ];

    let gemm_repeats = repeats.min(100);

    println!("\n=== GEMM (BF16 linear) ===");
    for &(m, k, n) in gemm_configs {
        bench(report, m, k, n, warmup.min(5), gemm_repeats)?;
    }
    Ok(())
}

fn bench(report: &mut BenchReport, m: usize, k: usize, n: usize, warmup: usize, repeats: usize) -> Result<()> {
    let device = Device::Cpu;

    let input_data: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.007) - 0.5).sin()))
        .collect();
    let weight_data: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(((i as f32 * 0.013) + 0.2).cos()))
        .collect();

    let input = Tensor::from_vec(input_data, (m, k), &device)?;
    let weight = Tensor::from_vec(weight_data, (n, k), &device)?;

    // custom cpu_ops GEMM
    let custom_us = {
        let in_storage = input.storage_and_layout();
        let w_storage = weight.storage_and_layout();
        let in_u16: &[u16] = match &*in_storage.0 {
            candle_core::Storage::Cpu(s) => {
                let sl = s.as_slice::<bf16>().unwrap();
                unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u16, sl.len()) }
            }
            _ => unreachable!(),
        };
        let w_u16: &[u16] = match &*w_storage.0 {
            candle_core::Storage::Cpu(s) => {
                let sl = s.as_slice::<bf16>().unwrap();
                unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u16, sl.len()) }
            }
            _ => unreachable!(),
        };
        let mut out_buf = vec![0u16; m * n];
        for _ in 0..warmup {
            prelude_cpu::ops::gemm::bf16_gemm_small_m(&mut out_buf, in_u16, w_u16, m, k, n);
        }
        let start = Instant::now();
        for _ in 0..repeats {
            prelude_cpu::ops::gemm::bf16_gemm_small_m(&mut out_buf, in_u16, w_u16, m, k, n);
        }
        start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0
    };

    // brgemm BF16 GEMM (oneDNN micro-kernel)
    let brgemm_us = {
        use prelude_cpu::onednn::{brgemm_available, BrgemmPackedWeight};
        if brgemm_available() {
            let weight_p: prelude_core::tensor::Tensor = weight.clone().into();
            let input_p: prelude_core::tensor::Tensor = input.clone().into();
            match BrgemmPackedWeight::pack(&weight_p)? {
                Some(brg_packed) => {
                    let brg = std::sync::Arc::new(brg_packed);
                    for _ in 0..warmup {
                        let _ = prelude_cpu::onednn::brgemm_gemm_forward_pub(&input_p, &brg, m, k, n)?;
                    }
                    let start = Instant::now();
                    for _ in 0..repeats {
                        let _ = prelude_cpu::onednn::brgemm_gemm_forward_pub(&input_p, &brg, m, k, n)?;
                    }
                    Some(start.elapsed().as_nanos() as f64 / repeats as f64 / 1000.0)
                }
                None => None,
            }
        } else {
            None
        }
    };

    let label = format!("{m}x{k}x{n}");
    let weight_mb = (n * k * 2) as f64 / 1e6;
    let custom_bw = weight_mb / (custom_us / 1e6) / 1e3;
    print!("  gemm [{label:<18}]  custom={custom_us:>10.1}us ({custom_bw:.0} GB/s)");
    if let Some(brg) = brgemm_us {
        let brg_bw = weight_mb / (brg / 1e6) / 1e3;
        print!("  brgemm={brg:>10.1}us ({brg_bw:.0} GB/s)");
    }
    println!();

    report.add(BenchEntry {
        category: "cpu/gemm".into(),
        name: format!("custom {label}"),
        ours_us: custom_us,
        baseline_name: None,
        baseline_us: None,
        note: Some(format!("{custom_bw:.0} GB/s")),
    });
    if let Some(brg) = brgemm_us {
        report.add(BenchEntry {
            category: "cpu/gemm".into(),
            name: format!("brgemm {label}"),
            ours_us: brg,
            baseline_name: Some("custom".into()),
            baseline_us: Some(custom_us),
            note: None,
        });
    }
    Ok(())
}
