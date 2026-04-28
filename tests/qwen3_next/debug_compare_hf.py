"""Compare HF model internals vs manual implementation at Layer 0 DeltaNet.
Identifies exactly WHERE the computation diverges.

Requires: transformers>=5.2.0, accelerate, safetensors, torch
Run on h200: python tests/qwen3_next/debug_compare_hf.py
"""
import torch
import torch.nn.functional as F
import json
import time
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_DIR = "/path/to/qwen3-next"


def load_raw_weights(keys_needed):
    """Load specific weights from safetensors."""
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        index = json.load(f)

    weights = {}
    shards_needed = set()
    for key in keys_needed:
        if key in index["weight_map"]:
            shards_needed.add(index["weight_map"][key])

    for shard in shards_needed:
        path = f"{MODEL_DIR}/{shard}"
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key in keys_needed:
                    weights[key] = f.get_tensor(key)
    return weights


def compare_tensors(name, a, b, atol=1e-6):
    """Compare two tensors and report differences."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH! {a.shape} vs {b.shape}")
        return False

    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if max_diff < atol:
        print(f"  {name}: MATCH (max_diff={max_diff:.2e}, shape={list(a.shape)})")
        return True
    else:
        print(f"  {name}: DIFFER! max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, shape={list(a.shape)}")
        print(f"    a: mean={a_f.mean():.6f} min={a_f.min():.6f} max={a_f.max():.6f}")
        print(f"    b: mean={b_f.mean():.6f} min={b_f.min():.6f} max={b_f.max():.6f}")
        # Show first few differing values
        flat_diff = diff.flatten()
        top_idx = flat_diff.argsort(descending=True)[:5]
        for idx in top_idx:
            i = idx.item()
            print(f"    [flat {i}]: a={a_f.flatten()[i]:.6f}, b={b_f.flatten()[i]:.6f}, diff={flat_diff[i]:.6f}")
        return False


def main():
    print("=" * 70)
    print("STEP 1: Compare weight values (HF model vs raw safetensors)")
    print("=" * 70)

    # Load HF model
    print("\nLoading HF model (bf16, CPU)...")
    t0 = time.time()
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"HF model loaded in {time.time()-t0:.1f}s")

    # Check layer types
    print(f"\nConfig layer_types (first 8): {config.layer_types[:8]}")
    print(f"Layer 0 type: {config.layer_types[0]}")

    # Access layer 0 attention module
    layer0 = model.model.layers[0]
    print(f"Layer 0 children: {[n for n,_ in layer0.named_children()]}")
    gdn = layer0.linear_attn
    print(f"\nLayer 0 attention class: {type(gdn).__name__}")
    print(f"  num_k_heads={gdn.num_k_heads}, num_v_heads={gdn.num_v_heads}")
    print(f"  head_k_dim={gdn.head_k_dim}, head_v_dim={gdn.head_v_dim}")
    print(f"  key_dim={gdn.key_dim}, value_dim={gdn.value_dim}")
    print(f"  conv_dim={gdn.conv_dim}, conv_kernel_size={gdn.conv_kernel_size}")

    # Check which libraries are available
    print(f"\n  causal_conv1d_fn available: {gdn.causal_conv1d_fn is not None}")
    print(f"  chunk_gated_delta_rule: {gdn.chunk_gated_delta_rule.__module__}")
    print(f"  recurrent_gated_delta_rule: {gdn.recurrent_gated_delta_rule.__module__}")

    # Load raw safetensors weights for layer 0
    raw_keys = [
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.linear_attn.in_proj_ba.weight",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
    ]
    print("\nLoading raw safetensors weights...")
    raw = load_raw_weights(set(raw_keys))

    # Compare weights
    print("\n--- Weight comparison (HF loaded vs raw safetensors) ---")
    weight_pairs = [
        ("in_proj_qkvz", gdn.in_proj_qkvz.weight.data,
         raw.get("model.layers.0.linear_attn.in_proj_qkvz.weight")),
        ("in_proj_ba", gdn.in_proj_ba.weight.data,
         raw.get("model.layers.0.linear_attn.in_proj_ba.weight")),
        ("conv1d", gdn.conv1d.weight.data,
         raw.get("model.layers.0.linear_attn.conv1d.weight")),
        ("dt_bias", gdn.dt_bias.data,
         raw.get("model.layers.0.linear_attn.dt_bias")),
        ("A_log", gdn.A_log.data,
         raw.get("model.layers.0.linear_attn.A_log")),
        ("norm", gdn.norm.weight.data,
         raw.get("model.layers.0.linear_attn.norm.weight")),
        ("out_proj", gdn.out_proj.weight.data,
         raw.get("model.layers.0.linear_attn.out_proj.weight")),
    ]

    all_match = True
    for name, hf_w, raw_w in weight_pairs:
        if raw_w is None:
            print(f"  {name}: RAW WEIGHT NOT FOUND!")
            all_match = False
            continue
        hf_w_cpu = hf_w.detach().cpu()
        raw_w_cpu = raw_w.detach().cpu()
        if not compare_tensors(name, hf_w_cpu, raw_w_cpu):
            all_match = False

    if all_match:
        print("\n>>> ALL WEIGHTS MATCH <<<")
    else:
        print("\n>>> WEIGHT MISMATCH DETECTED! <<<")

    print("\n" + "=" * 70)
    print("STEP 2: Compare intermediate computation at Layer 0")
    print("=" * 70)

    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    print(f"\nInput IDs: {input_ids}")
    input_tensor = torch.tensor([input_ids])

    # Hook layer 0 DeltaNet to capture intermediates
    captured = {}

    original_forward = type(gdn).forward

    def hooked_forward(self_gdn, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        """Monkey-patched forward that captures intermediate values."""
        batch_size, seq_len, _ = hidden_states.shape

        captured["input"] = hidden_states.detach().clone()

        # QKVZ projection
        projected_states_qkvz = self_gdn.in_proj_qkvz(hidden_states)
        projected_states_ba = self_gdn.in_proj_ba(hidden_states)
        captured["proj_qkvz"] = projected_states_qkvz.detach().clone()
        captured["proj_ba"] = projected_states_ba.detach().clone()

        # Interleaved split
        query, key, value, z, b, a = self_gdn.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba)
        captured["split_q"] = query.detach().clone()  # (B, L, num_k_heads, head_k_dim)
        captured["split_k"] = key.detach().clone()
        captured["split_v"] = value.detach().clone()  # (B, L, num_v_heads, head_v_dim)
        captured["split_z"] = z.detach().clone()
        captured["split_b"] = b.detach().clone()
        captured["split_a"] = a.detach().clone()

        # Flatten Q, K, V for conv1d
        query_flat, key_flat, value_flat = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
        mixed_qkv = torch.cat((query_flat, key_flat, value_flat), dim=-1)
        captured["conv_input"] = mixed_qkv.detach().clone()  # (B, L, conv_dim)

        mixed_qkv_t = mixed_qkv.transpose(1, 2)  # (B, conv_dim, L)

        # Conv1d (use whatever path is available)
        if self_gdn.causal_conv1d_fn is not None:
            conv_out = self_gdn.causal_conv1d_fn(
                x=mixed_qkv_t,
                weight=self_gdn.conv1d.weight.squeeze(1),
                bias=self_gdn.conv1d.bias,
                activation=self_gdn.activation,
                seq_idx=None,
            )
        else:
            conv_out = F.silu(self_gdn.conv1d(mixed_qkv_t)[:, :, :seq_len])

        captured["conv_output_raw"] = conv_out.detach().clone()  # (B, conv_dim, L)

        conv_out = conv_out.transpose(1, 2)  # (B, L, conv_dim)

        # Split back
        q_post, k_post, v_post = torch.split(
            conv_out, [self_gdn.key_dim, self_gdn.key_dim, self_gdn.value_dim], dim=-1)
        q_post = q_post.reshape(q_post.shape[0], q_post.shape[1], -1, self_gdn.head_k_dim)
        k_post = k_post.reshape(k_post.shape[0], k_post.shape[1], -1, self_gdn.head_k_dim)
        v_post = v_post.reshape(v_post.shape[0], v_post.shape[1], -1, self_gdn.head_v_dim)
        captured["post_conv_q"] = q_post.detach().clone()
        captured["post_conv_k"] = k_post.detach().clone()
        captured["post_conv_v"] = v_post.detach().clone()

        # Gating
        beta = b.sigmoid()
        g = -self_gdn.A_log.float().exp() * F.softplus(a.float() + self_gdn.dt_bias)
        captured["beta"] = beta.detach().clone()
        captured["g"] = g.detach().clone()

        # Repeat interleave
        kv_ratio = self_gdn.num_v_heads // self_gdn.num_k_heads
        if kv_ratio > 1:
            q_post = q_post.repeat_interleave(kv_ratio, dim=2)
            k_post = k_post.repeat_interleave(kv_ratio, dim=2)
        captured["expanded_q"] = q_post.detach().clone()
        captured["expanded_k"] = k_post.detach().clone()

        # Call actual recurrence
        core_attn_out, _ = self_gdn.chunk_gated_delta_rule(
            q_post, k_post, v_post, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        captured["recurrence_out"] = core_attn_out.detach().clone()

        # Norm
        z_shape_og = z.shape
        core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        normed = self_gdn.norm(core_attn_out_flat, z_flat)
        normed = normed.reshape(z_shape_og)
        normed = normed.reshape(normed.shape[0], normed.shape[1], -1)
        captured["normed"] = normed.detach().clone()

        # Out proj
        output = self_gdn.out_proj(normed)
        captured["output"] = output.detach().clone()

        return output

    # Monkey-patch
    import types
    gdn.forward = types.MethodType(hooked_forward, gdn)

    # Run HF forward
    print("\nRunning HF model forward pass...")
    with torch.no_grad():
        hf_output = model(input_tensor)

    hf_logits = hf_output.logits[0, -1].float()
    top5 = hf_logits.argsort(descending=True)[:5]
    print(f"\nHF Top-5:")
    for i, idx in enumerate(top5):
        print(f"  #{i+1}: id={idx.item()} val={hf_logits[idx].item():.4f} text={repr(tokenizer.decode([idx.item()]))}")

    # Restore original forward
    gdn.forward = types.MethodType(original_forward, gdn)

    print("\n" + "=" * 70)
    print("STEP 3: Manual computation and comparison")
    print("=" * 70)

    # Use captured input to layer 0 DeltaNet
    h = captured["input"]
    B, L, hidden_dim = h.shape
    print(f"\nInput to layer 0 DeltaNet: shape={list(h.shape)}")
    print(f"  mean={h.float().mean():.6f} min={h.float().min():.6f} max={h.float().max():.6f}")

    # Load raw weights
    qkvz_w = raw["model.layers.0.linear_attn.in_proj_qkvz.weight"].to(torch.bfloat16)
    ba_w = raw["model.layers.0.linear_attn.in_proj_ba.weight"].to(torch.bfloat16)
    conv_w = raw["model.layers.0.linear_attn.conv1d.weight"].to(torch.bfloat16)
    dt_bias = raw["model.layers.0.linear_attn.dt_bias"]
    A_log = raw["model.layers.0.linear_attn.A_log"]
    norm_w = raw["model.layers.0.linear_attn.norm.weight"].to(torch.bfloat16)
    out_proj_w = raw["model.layers.0.linear_attn.out_proj.weight"].to(torch.bfloat16)

    num_k_heads = 16
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    key_dim = num_k_heads * head_k_dim  # 2048
    value_dim = num_v_heads * head_v_dim  # 4096
    conv_dim = key_dim * 2 + value_dim  # 8192
    kv_ratio = num_v_heads // num_k_heads  # 2
    conv_kernel = 4

    # Step A: QKVZ projection
    print("\n--- A. QKVZ Projection ---")
    manual_qkvz = F.linear(h, qkvz_w)
    manual_ba = F.linear(h, ba_w)
    compare_tensors("proj_qkvz", captured["proj_qkvz"], manual_qkvz)
    compare_tensors("proj_ba", captured["proj_ba"], manual_ba)

    # Step B: Interleaved split
    print("\n--- B. Interleaved Split ---")
    group_dim = head_k_dim * 2 + kv_ratio * head_v_dim * 2  # 768
    fp_g = manual_qkvz.view(B, L, num_k_heads, group_dim)

    m_q = fp_g[..., :head_k_dim]  # (B, L, 16, 128)
    m_k = fp_g[..., head_k_dim:2*head_k_dim]  # (B, L, 16, 128)
    v_off = 2 * head_k_dim
    m_v = fp_g[..., v_off:v_off+kv_ratio*head_v_dim]  # (B, L, 16, 256)
    m_v = m_v.reshape(B, L, num_v_heads, head_v_dim)  # (B, L, 32, 128) — match HF reshape
    m_z = fp_g[..., v_off+kv_ratio*head_v_dim:]  # (B, L, 16, 256)
    m_z = m_z.reshape(B, L, num_v_heads, head_v_dim)  # (B, L, 32, 128)

    compare_tensors("split_q", captured["split_q"], m_q)
    compare_tensors("split_k", captured["split_k"], m_k)
    compare_tensors("split_v", captured["split_v"], m_v)
    compare_tensors("split_z", captured["split_z"], m_z)

    # BA split
    ba_g = manual_ba.view(B, L, num_k_heads, kv_ratio * 2)  # (B, L, 16, 4)
    m_b = ba_g[..., :kv_ratio].reshape(B, L, num_v_heads)  # (B, L, 32)
    m_a = ba_g[..., kv_ratio:].reshape(B, L, num_v_heads)  # (B, L, 32)
    compare_tensors("split_b", captured["split_b"], m_b)
    compare_tensors("split_a", captured["split_a"], m_a)

    # Step C: Conv input (flatten and concat)
    print("\n--- C. Conv Input ---")
    m_q_flat = m_q.reshape(B, L, key_dim)
    m_k_flat = m_k.reshape(B, L, key_dim)
    m_v_flat = m_v.reshape(B, L, value_dim)
    m_conv_input = torch.cat([m_q_flat, m_k_flat, m_v_flat], dim=-1)
    compare_tensors("conv_input", captured["conv_input"], m_conv_input)

    # Step D: Conv1d output
    print("\n--- D. Conv1d ---")
    m_conv_input_t = m_conv_input.transpose(1, 2)  # (B, conv_dim, L)

    # Manual causal conv: left-pad with zeros
    pad = torch.zeros(B, conv_dim, conv_kernel - 1, dtype=torch.bfloat16)
    padded = torch.cat([pad, m_conv_input_t], dim=2)  # (B, conv_dim, L+3)
    m_conv_out = F.conv1d(padded.float(), conv_w.float(), padding=0, groups=conv_dim)
    m_conv_out = m_conv_out.to(torch.bfloat16)  # (B, conv_dim, L)

    # Apply SiLU
    m_conv_out_silu = F.silu(m_conv_out)

    # HF captured is already after SiLU (causal_conv1d_fn fuses activation)
    compare_tensors("conv_output (after SiLU)", captured["conv_output_raw"], m_conv_out_silu)

    # Also try WITHOUT SiLU to see if SiLU is the issue
    compare_tensors("conv_output (before SiLU)", captured["conv_output_raw"], m_conv_out)

    # Step E: Split post-conv
    print("\n--- E. Post-conv split ---")
    m_conv_t = m_conv_out_silu.transpose(1, 2)  # (B, L, conv_dim)
    m_qc = m_conv_t[..., :key_dim].reshape(B, L, num_k_heads, head_k_dim)
    m_kc = m_conv_t[..., key_dim:2*key_dim].reshape(B, L, num_k_heads, head_k_dim)
    m_vc = m_conv_t[..., 2*key_dim:].reshape(B, L, num_v_heads, head_v_dim)
    compare_tensors("post_conv_q", captured["post_conv_q"], m_qc)
    compare_tensors("post_conv_k", captured["post_conv_k"], m_kc)
    compare_tensors("post_conv_v", captured["post_conv_v"], m_vc)

    # Step F: Gating
    print("\n--- F. Gating ---")
    m_beta = m_b.sigmoid()
    m_g = -A_log.float().exp() * F.softplus(m_a.float() + dt_bias)
    compare_tensors("beta", captured["beta"], m_beta)
    compare_tensors("g", captured["g"], m_g)

    # Step G: Repeat interleave
    print("\n--- G. Repeat interleave ---")
    m_qc_exp = m_qc.repeat_interleave(kv_ratio, dim=2)  # (B, L, 32, 128)
    m_kc_exp = m_kc.repeat_interleave(kv_ratio, dim=2)
    compare_tensors("expanded_q", captured["expanded_q"], m_qc_exp)
    compare_tensors("expanded_k", captured["expanded_k"], m_kc_exp)

    # Step H: Recurrence (manual per-token loop)
    print("\n--- H. Recurrence (manual vs HF) ---")

    # Manual recurrence matching HF's torch_chunk/recurrent implementation
    def manual_recurrence(q, k, v, g, beta):
        """Manual per-token recurrence matching HF's torch_recurrent_gated_delta_rule."""
        from transformers.models.qwen3_next.modeling_qwen3_next import l2norm

        # L2 normalize
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

        # transpose to (B, heads, L, dim) and convert to float
        q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

        B, num_heads, seq_len, k_dim = k.shape
        v_dim = v.shape[-1]
        scale = 1.0 / (k_dim ** 0.5)
        q = q * scale

        out = torch.zeros(B, num_heads, seq_len, v_dim, dtype=torch.float32)
        state = torch.zeros(B, num_heads, k_dim, v_dim, dtype=torch.float32)

        for i in range(seq_len):
            q_t = q[:, :, i]
            k_t = k[:, :, i]
            v_t = v[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        return out.transpose(1, 2).contiguous().to(torch.bfloat16)

    # m_vc stays in (B, L, num_v_heads, head_v_dim) shape
    # need beta and g with proper shape (B, L, num_v_heads)
    m_beta_3d = m_beta.unsqueeze(-1)  # (B, L, 32, 1)... no wait
    # Actually beta is (B, L, 32), g is (B, L, 32)
    # We need them as (B, L, 32, 1) for the recurrence? No.
    # The HF code passes beta as (B, L, num_v_heads) to the recurrence
    # which transposes to (B, num_v_heads, L) and the recurrence indexes per-step

    m_rec_out = manual_recurrence(m_qc_exp, m_kc_exp, m_vc, m_g, m_beta)
    compare_tensors("recurrence_out", captured["recurrence_out"], m_rec_out)

    # Also print value ranges
    print(f"\n  HF recurrence out: mean={captured['recurrence_out'].float().mean():.6f} "
          f"min={captured['recurrence_out'].float().min():.6f} max={captured['recurrence_out'].float().max():.6f}")
    print(f"  Manual recurrence out: mean={m_rec_out.float().mean():.6f} "
          f"min={m_rec_out.float().min():.6f} max={m_rec_out.float().max():.6f}")

    # Step I: Gated norm
    print("\n--- I. Gated RMSNorm ---")
    m_z_flat = m_z.reshape(B * L, num_v_heads, head_v_dim)
    m_rec_flat = m_rec_out.reshape(B * L, num_v_heads, head_v_dim)

    # RMS norm per head
    x_f32 = m_rec_flat.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + 1e-6)
    normed = (normed * norm_w.float()).to(torch.bfloat16)

    # Gate with SiLU(z)
    z_gate = F.silu(m_z_flat.float()).to(torch.bfloat16)
    m_normed = (normed * z_gate).reshape(B, L, value_dim)
    compare_tensors("normed", captured["normed"], m_normed)

    # Step J: Output projection
    print("\n--- J. Output projection ---")
    m_output = F.linear(m_normed, out_proj_w)
    compare_tensors("output", captured["output"], m_output)

    print(f"\n  HF output: mean={captured['output'].float().mean():.6f} "
          f"min={captured['output'].float().min():.6f} max={captured['output'].float().max():.6f}")
    print(f"  Manual output: mean={m_output.float().mean():.6f} "
          f"min={m_output.float().min():.6f} max={m_output.float().max():.6f}")

    print("\n" + "=" * 70)
    print("STEP 4: Compare manual output with our debug_full_forward.py approach")
    print("=" * 70)

    # Now reproduce EXACTLY what our debug_full_forward.py does (our buggy approach)
    # and compare with the manual recurrence above
    print("\n--- Reproducing debug_full_forward.py deltanet logic ---")

    def our_l2_normalize(x, eps=1e-6):
        norm = torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)
        return x / norm

    def our_softplus(x, beta_coeff=1.0, threshold=20.0):
        mask = (beta_coeff * x) <= threshold
        safe = torch.where(mask, x, torch.zeros_like(x))
        sp = (1.0 / beta_coeff) * torch.log1p(torch.exp(beta_coeff * safe))
        return torch.where(mask, sp, x)

    # Our script's approach: split, conv, then per-token loop
    # Use same split results
    our_q = m_q.reshape(B, L, key_dim)
    our_k = m_k.reshape(B, L, key_dim)
    our_v = m_v.reshape(B, L, value_dim)
    our_z = m_z.reshape(B, L, value_dim)

    # Conv (same as above)
    our_qkv = torch.cat([our_q, our_k, our_v], dim=-1)
    our_qkv_t = our_qkv.transpose(1, 2)
    our_pad = torch.zeros(B, conv_dim, conv_kernel - 1, dtype=torch.bfloat16)
    our_padded = torch.cat([our_pad, our_qkv_t], dim=2)
    our_conv_out = F.conv1d(our_padded.float(), conv_w.float(), padding=0, groups=conv_dim)
    our_conv_out = our_conv_out.to(torch.bfloat16).transpose(1, 2)
    our_conv_out = F.silu(our_conv_out)

    our_qc = our_conv_out[..., :key_dim]
    our_kc = our_conv_out[..., key_dim:2*key_dim]
    our_vc = our_conv_out[..., 2*key_dim:]

    # Our per-token recurrence
    scale = head_k_dim ** -0.5
    state = torch.zeros(num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
    our_outputs = []

    for t in range(L):
        qt = our_qc[0, t].reshape(num_k_heads, head_k_dim).float()
        kt = our_kc[0, t].reshape(num_k_heads, head_k_dim).float()
        vt = our_vc[0, t].reshape(num_v_heads, head_v_dim).float()
        bt = m_b[0, t].float()
        at = m_a[0, t].float()

        qt = our_l2_normalize(qt)
        kt = our_l2_normalize(kt)
        k_exp = kt.repeat_interleave(kv_ratio, dim=0)
        q_exp = qt.repeat_interleave(kv_ratio, dim=0)
        q_exp = q_exp * scale

        g_t = -A_log.float().exp() * our_softplus(at + dt_bias.float())
        beta_t = bt.sigmoid()

        state = state * g_t.exp().unsqueeze(1).unsqueeze(2)
        prediction = torch.bmm(state.transpose(1, 2), k_exp.unsqueeze(2)).squeeze(2)
        v_residual = vt - prediction
        v_gated = v_residual * beta_t.unsqueeze(1)
        state = state + torch.bmm(k_exp.unsqueeze(2), v_gated.unsqueeze(1))
        output = torch.bmm(state.transpose(1, 2), q_exp.unsqueeze(2)).squeeze(2)
        our_outputs.append(output)

    our_rec_out = torch.stack(our_outputs, 0).to(torch.bfloat16)
    our_rec_out = our_rec_out.reshape(1, L, value_dim)

    print(f"  Our recurrence out: mean={our_rec_out.float().mean():.6f} "
          f"min={our_rec_out.float().min():.6f} max={our_rec_out.float().max():.6f}")
    print(f"  HF recurrence out:  mean={captured['recurrence_out'].float().mean():.6f} "
          f"min={captured['recurrence_out'].float().min():.6f} max={captured['recurrence_out'].float().max():.6f}")

    # Reshape for comparison
    our_rec_heads = our_rec_out.reshape(B, L, num_v_heads, head_v_dim)
    hf_rec_heads = captured["recurrence_out"]
    compare_tensors("our_recurrence vs HF", hf_rec_heads, our_rec_heads)

    if not torch.allclose(hf_rec_heads.float(), our_rec_heads.float(), atol=0.01):
        print("\n  *** RECURRENCE OUTPUT DIFFERS! Investigating per-token... ***")
        for t in range(L):
            hf_t = hf_rec_heads[0, t].float()
            our_t = our_rec_heads[0, t].float()
            diff = (hf_t - our_t).abs().max().item()
            print(f"    Token {t}: max_diff={diff:.6f}, "
                  f"hf_max={hf_t.abs().max():.6f}, our_max={our_t.abs().max():.6f}, "
                  f"ratio={hf_t.abs().max()/(our_t.abs().max()+1e-10):.2f}x")


if __name__ == "__main__":
    main()
