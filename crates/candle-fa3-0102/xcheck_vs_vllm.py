#!/usr/bin/env python3
"""Cross-check candle-fa3-0102 vs vLLM-FA3 vs an fp32 eager reference on IDENTICAL
random bf16 inputs. Answers: do candle and vLLM produce the same output?

Flow: gen random bf16 q/k/v -> save safetensors -> run vLLM FA3 -> shell out to the
candle XCHECK example (reads same safetensors, writes its output) -> load both ->
compare against each other and the fp32 reference.
Run in vllm env: CUDA_VISIBLE_DEVICES=0 python xcheck_vs_vllm.py
"""
import os, subprocess
import torch
from safetensors.torch import save_file, load_file

DEV = "cuda:0"
S, HQ, HKV, D = 512, 32, 8, 128
CAUSAL = True
QKV = "/tmp/fa_qkv.safetensors"
COUT = "/tmp/fa_candle_out.safetensors"
scale = 1.0 / (D ** 0.5)
torch.manual_seed(0)

q = torch.randn(S, HQ, D, device=DEV, dtype=torch.bfloat16)
k = torch.randn(S, HKV, D, device=DEV, dtype=torch.bfloat16)
v = torch.randn(S, HKV, D, device=DEV, dtype=torch.bfloat16)
save_file({"q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous()}, QKV)
print(f"inputs: S={S} HQ={HQ} HKV={HKV} D={D} causal={CAUSAL} bf16 -> {QKV}")

# ---- vLLM FA3 ----
from vllm.vllm_flash_attn import flash_attn_varlen_func
cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
vllm_out = flash_attn_varlen_func(q, k, v, max_seqlen_q=S, cu_seqlens_q=cu,
                                  max_seqlen_k=S, cu_seqlens_k=cu,
                                  softmax_scale=scale, causal=CAUSAL, fa_version=3).float()  # (S,HQ,D)

# ---- fp32 eager reference (GQA expand + causal) ----
rep = HQ // HKV
kf = k.float().repeat_interleave(rep, dim=1)  # (S,HQ,D)
vf = v.float().repeat_interleave(rep, dim=1)
qf = q.float()
scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale  # (HQ,S,S)
if CAUSAL:
    mask = torch.triu(torch.ones(S, S, device=DEV, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
attn = torch.softmax(scores, dim=-1)
ref = torch.einsum("hqk,khd->qhd", attn, vf)  # (S,HQ,D)

# ---- candle FA3 (shell out to the Rust XCHECK example) ----
PRE = "/scratch/xueying/miniforge3/envs/prelude"
CUDA = "/usr/local/cuda-12.9"
env = dict(os.environ)
env["PATH"] = f"{CUDA}/bin:{PRE}/bin:" + env.get("PATH", "")
env["LD_LIBRARY_PATH"] = f"{CUDA}/targets/x86_64-linux/lib:{PRE}/lib:" + env.get("LD_LIBRARY_PATH", "")
env["CUDA_VISIBLE_DEVICES"] = "0"
env["XCHECK"] = "1"; env["CAUSAL"] = "1" if CAUSAL else "0"; env["GQA_PACK"] = os.environ.get("XPACK","0")
env["XVARLEN"] = os.environ.get("XVARLEN","0")
env["XCHECK_IN"] = QKV; env["XCHECK_OUT"] = COUT
crate = "/data/xueying/benchmark/prelude-fa3-0102/crates/candle-fa3-0102"
r = subprocess.run([f"{crate}/target/release/examples/bench_fa3_0102"], env=env,
                   capture_output=True, text=True)
print("candle:", r.stdout.strip(), r.stderr.strip()[-200:] if r.returncode else "")
assert r.returncode == 0, "candle XCHECK failed"
candle_out = load_file(COUT)["out"].to(DEV).float()  # (S,HQ,D)

def cmp(name, a, b):
    d = (a - b).abs()
    denom = b.abs().clamp_min(1e-3)
    print(f"{name:24}: max_abs={d.max():.4e}  mean_abs={d.mean():.4e}  "
          f"max_rel={(d/denom).max():.4e}  cos={torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0):.6f}")

print(f"\nshapes: candle={tuple(candle_out.shape)} vllm={tuple(vllm_out.shape)} ref={tuple(ref.shape)}")
cmp("candle  vs vLLM", candle_out, vllm_out)
cmp("candle  vs ref(fp32)", candle_out, ref)
cmp("vLLM    vs ref(fp32)", vllm_out, ref)
