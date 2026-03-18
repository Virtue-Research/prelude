<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="Prelude" height="80">
  </picture>
</p>

<p align="center">
  Fast LLM inference engine in Rust. Optimized for prefill throughput.
</p>

---

## Performance

**GPU (H200, Qwen3-0.6B, BF16)**

| Task           | Concurrency |  Throughput   | Latency P95 |
|----------------|:-----------:|:-------------:|:-----------:|
| Generation     |      8      |  1,768 tok/s  |    138ms    |
| Generation     |     16      |  1,822 tok/s  |    285ms    |
| Classification |     16      | 2,060 items/s |    166ms    |
| Embedding      |     16      | 1,983 items/s |    168ms    |

TPOT (time per output token): **3.2ms** at c=1, **3.8ms** at c=16.

**CPU (Xeon 8480+, Qwen3-0.6B, BF16 via oneDNN)**

| Benchmark              |   Prelude   |   SGLang   |  Speedup  |
|------------------------|:-----------:|:----------:|:---------:|
| Prefill (128 tok, c=1) |  3,629 t/s  | 2,298 t/s  | **1.58x** |
| Prefill (128 tok, c=4) |  5,710 t/s  | 3,970 t/s  | **1.44x** |
| Decode (32 in, 32 out) | 109.6 out/s | 57.1 out/s | **1.92x** |

## License

Apache-2.0
