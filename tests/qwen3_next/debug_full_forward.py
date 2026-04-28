"""Full forward pass through all 48 layers to compare with Rust."""
import torch
import torch.nn.functional as F
import json
import time
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = "/path/to/qwen3-next"

def load_all_weights():
    """Load all weights from safetensors into memory."""
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        index = json.load(f)

    # Find unique shards
    shards = sorted(set(index["weight_map"].values()))

    weights = {}
    for i, shard in enumerate(shards):
        t0 = time.time()
        path = f"{MODEL_DIR}/{shard}"
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                # Skip MTP weights
                if key.startswith("mtp."):
                    continue
                weights[key] = f.get_tensor(key)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(shards)} shards ({time.time()-t0:.1f}s)")

    print(f"Total weights loaded: {len(weights)}")
    return weights


def rms_norm(x, weight, eps=1e-6, residual=True):
    """RMSNorm. Qwen3-Next uses residual parameterization (1+weight) for most norms,
    but the DeltaNet gated norm uses standard weight."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    w = (1.0 + weight.float()) if residual else weight.float()
    return (x_f32 * torch.rsqrt(variance + eps) * w).to(x.dtype)


def l2_normalize(x, eps=1e-6):
    norm = torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)
    return x / norm


def softplus(x, beta=1.0, threshold=20.0):
    mask = (beta * x) <= threshold
    safe = torch.where(mask, x, torch.zeros_like(x))
    sp = (1.0 / beta) * torch.log1p(torch.exp(beta * safe))
    return torch.where(mask, sp, x)


def apply_partial_rope(q, k, cos, sin, rotary_dim, offset):
    """Apply RoPE to first rotary_dim dimensions only."""
    # q, k: [B, H, L, D]
    seq_len = q.shape[2]
    cos_t = cos[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, L, rotary_dim/2]
    sin_t = sin[offset:offset+seq_len].unsqueeze(0).unsqueeze(0)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    half = rotary_dim // 2
    q1, q2 = q_rot[..., :half], q_rot[..., half:]
    k1, k2 = k_rot[..., :half], k_rot[..., half:]

    q_rot = torch.cat([q1 * cos_t - q2 * sin_t, q2 * cos_t + q1 * sin_t], dim=-1)
    k_rot = torch.cat([k1 * cos_t - k2 * sin_t, k2 * cos_t + k1 * sin_t], dim=-1)

    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def deltanet_forward(h, weights, prefix, cfg, layer_states):
    """Forward pass for Gated DeltaNet layer."""
    B, L, _ = h.shape
    num_k_heads = cfg["linear_num_key_heads"]
    num_v_heads = cfg["linear_num_value_heads"]
    head_k_dim = cfg["linear_key_head_dim"]
    head_v_dim = cfg["linear_value_head_dim"]
    conv_kernel = cfg["linear_conv_kernel_dim"]
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    kv_ratio = num_v_heads // num_k_heads

    # Project
    qkvz_w = weights[f"{prefix}.in_proj_qkvz.weight"].to(torch.bfloat16)
    ba_w = weights[f"{prefix}.in_proj_ba.weight"].to(torch.bfloat16)
    full_proj = F.linear(h, qkvz_w)
    ba = F.linear(h, ba_w)

    # Interleaved QKVZ split
    group_dim = head_k_dim * 2 + kv_ratio * head_v_dim * 2
    fp_g = full_proj.view(B, L, num_k_heads, group_dim)
    q = fp_g[..., :head_k_dim].reshape(B, L, key_dim)
    k = fp_g[..., head_k_dim:2*head_k_dim].reshape(B, L, key_dim)
    v_off = 2 * head_k_dim
    v = fp_g[..., v_off:v_off+kv_ratio*head_v_dim].reshape(B, L, value_dim)
    z = fp_g[..., v_off+kv_ratio*head_v_dim:].reshape(B, L, value_dim)

    # BA split
    ba_g = ba.view(B, L, num_k_heads, kv_ratio * 2)
    b_param = ba_g[..., :kv_ratio].reshape(B, L, num_v_heads)
    a_param = ba_g[..., kv_ratio:].reshape(B, L, num_v_heads)

    # Conv1d
    conv_w = weights[f"{prefix}.conv1d.weight"].to(torch.bfloat16)
    qkv = torch.cat([q, k, v], dim=-1)
    qkv_t = qkv.transpose(1, 2)
    pad = torch.zeros(B, conv_dim, conv_kernel - 1, dtype=torch.bfloat16)
    padded = torch.cat([pad, qkv_t], dim=2)
    conv_out = F.conv1d(padded.float(), conv_w.float(), padding=0, stride=1, dilation=1, groups=conv_dim)
    conv_out = conv_out.to(torch.bfloat16).transpose(1, 2)
    conv_out = F.silu(conv_out)

    q_c = conv_out[..., :key_dim]
    k_c = conv_out[..., key_dim:2*key_dim]
    v_c = conv_out[..., 2*key_dim:]

    # Delta rule recurrence
    dt_bias = weights[f"{prefix}.dt_bias"]
    A_log = weights[f"{prefix}.A_log"]
    scale = head_k_dim ** -0.5

    state = torch.zeros(num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
    outputs = []

    for t in range(L):
        qt = q_c[0, t].reshape(num_k_heads, head_k_dim).float()
        kt = k_c[0, t].reshape(num_k_heads, head_k_dim).float()
        vt = v_c[0, t].reshape(num_v_heads, head_v_dim).float()
        bt = b_param[0, t].float()
        at = a_param[0, t].float()

        qt = l2_normalize(qt)
        kt = l2_normalize(kt)
        k_exp = kt.repeat_interleave(kv_ratio, dim=0)
        q_exp = qt.repeat_interleave(kv_ratio, dim=0)
        q_exp = q_exp * scale

        g = -A_log.float().exp() * softplus(at + dt_bias.float())
        beta = bt.sigmoid()

        state = state * g.exp().unsqueeze(1).unsqueeze(2)
        prediction = torch.bmm(state.transpose(1, 2), k_exp.unsqueeze(2)).squeeze(2)
        v_residual = vt - prediction
        v_gated = v_residual * beta.unsqueeze(1)
        state = state + torch.bmm(k_exp.unsqueeze(2), v_gated.unsqueeze(1))
        output = torch.bmm(state.transpose(1, 2), q_exp.unsqueeze(2)).squeeze(2)
        outputs.append(output)

    output_stack = torch.stack(outputs, 0).to(torch.bfloat16)
    output_flat = output_stack.reshape(L, value_dim)

    # Gated RMSNorm
    norm_w = weights[f"{prefix}.norm.weight"].to(torch.bfloat16)
    o_h = output_flat.reshape(L, num_v_heads, head_v_dim)
    z_h = z.squeeze(0).reshape(L, num_v_heads, head_v_dim)
    normed = rms_norm(o_h, norm_w, eps=cfg["rms_norm_eps"], residual=False)
    gated = normed * F.silu(z_h)
    gated_flat = gated.reshape(L, value_dim)

    # Output projection
    out_proj_w = weights[f"{prefix}.out_proj.weight"].to(torch.bfloat16)
    return F.linear(gated_flat, out_proj_w).unsqueeze(0)


def attention_forward(h, weights, prefix, cfg, cos, sin, kv_cache, offset):
    """Forward pass for Gated Attention layer."""
    B, L, _ = h.shape
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
    kv_ratio = num_heads // num_kv_heads

    q_w = weights[f"{prefix}.q_proj.weight"].to(torch.bfloat16)
    k_w = weights[f"{prefix}.k_proj.weight"].to(torch.bfloat16)
    v_w = weights[f"{prefix}.v_proj.weight"].to(torch.bfloat16)
    o_w = weights[f"{prefix}.o_proj.weight"].to(torch.bfloat16)

    # HF splits Q and gate per-head: reshape to [B, L, H, 2*D] then chunk
    q_and_gate = F.linear(h, q_w).reshape(B, L, num_heads, head_dim * 2)
    q = q_and_gate[..., :head_dim]                             # [B, L, H, D]
    gate = q_and_gate[..., head_dim:].reshape(B, L, -1)       # [B, L, H*D]
    k = F.linear(h, k_w)
    v = F.linear(h, v_w)

    q = q.transpose(1, 2)  # already [B, L, H, D]
    k = k.reshape(B, L, num_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, L, num_kv_heads, head_dim).transpose(1, 2)

    # QK norm
    qn_w = weights[f"{prefix}.q_norm.weight"].to(torch.bfloat16)
    kn_w = weights[f"{prefix}.k_norm.weight"].to(torch.bfloat16)
    q = rms_norm(q.flatten(0, 2), qn_w, cfg["rms_norm_eps"]).reshape(B, num_heads, L, head_dim)
    k = rms_norm(k.flatten(0, 2), kn_w, cfg["rms_norm_eps"]).reshape(B, num_kv_heads, L, head_dim)

    # Partial RoPE
    q, k = apply_partial_rope(q, k, cos, sin, rotary_dim, offset)

    # KV cache
    if kv_cache is not None:
        prev_k, prev_v = kv_cache
        k = torch.cat([prev_k, k], dim=2)
        v = torch.cat([prev_v, v], dim=2)
    new_cache = (k.clone(), v.clone())

    # GQA expand
    if kv_ratio > 1:
        k = k.unsqueeze(2).expand(-1, -1, kv_ratio, -1, -1).reshape(B, num_heads, -1, head_dim)
        v = v.unsqueeze(2).expand(-1, -1, kv_ratio, -1, -1).reshape(B, num_heads, -1, head_dim)

    # Attention
    scale = head_dim ** -0.5
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    attn = (q_f @ k_f.transpose(-2, -1)) * scale

    kv_len = k.shape[2]
    if L > 1:
        mask = torch.full((L, kv_len), float('-inf'))
        off = kv_len - L
        for i in range(L):
            mask[i, :i+off+1] = 0.0
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

    attn = F.softmax(attn, dim=-1)
    out = (attn @ v_f).to(torch.bfloat16)
    out = out.transpose(1, 2).reshape(B, L, num_heads * head_dim)

    # Gate
    gate = torch.sigmoid(gate)
    out = out * gate

    return F.linear(out, o_w), new_cache


def moe_forward(h, weights, prefix, cfg):
    """Forward pass for Sparse MoE block."""
    B, L, hidden_dim = h.shape
    xs = h.reshape(-1, hidden_dim)

    # Shared expert
    se_gw = weights[f"{prefix}.shared_expert.gate_proj.weight"].to(torch.bfloat16)
    se_uw = weights[f"{prefix}.shared_expert.up_proj.weight"].to(torch.bfloat16)
    se_dw = weights[f"{prefix}.shared_expert.down_proj.weight"].to(torch.bfloat16)
    gate = F.silu(F.linear(xs, se_gw))
    up = F.linear(xs, se_uw)
    shared_out = F.linear(gate * up, se_dw)

    seg_w = weights[f"{prefix}.shared_expert_gate.weight"].to(torch.bfloat16)
    shared_gate = torch.sigmoid(F.linear(xs, seg_w))
    shared_contribution = shared_out * shared_gate

    # Router
    gate_w = weights[f"{prefix}.gate.weight"].to(torch.bfloat16)
    router_logits = F.linear(xs, gate_w)
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_vals, topk_ids = torch.topk(routing_weights, cfg["num_experts_per_tok"], dim=-1)
    topk_vals = topk_vals.float()
    if cfg["norm_topk_prob"]:
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    # Sequential expert routing
    routed = torch.zeros_like(xs)
    topk_ids_np = topk_ids.cpu().numpy()
    topk_vals_np = topk_vals.cpu().numpy()

    for expert_id in range(cfg["num_experts"]):
        mask = (topk_ids_np == expert_id)
        token_positions = mask.any(axis=1).nonzero()[0]
        if len(token_positions) == 0:
            continue

        e_gw = weights[f"{prefix}.experts.{expert_id}.gate_proj.weight"].to(torch.bfloat16)
        e_uw = weights[f"{prefix}.experts.{expert_id}.up_proj.weight"].to(torch.bfloat16)
        e_dw = weights[f"{prefix}.experts.{expert_id}.down_proj.weight"].to(torch.bfloat16)

        x_sub = xs[token_positions]
        e_gate = F.silu(F.linear(x_sub, e_gw))
        e_up = F.linear(x_sub, e_uw)
        e_out = F.linear(e_gate * e_up, e_dw)

        for j, tid in enumerate(token_positions):
            w = topk_vals_np[tid][mask[tid]].sum()
            routed[tid] += (e_out[j] * w).to(routed.dtype)

    result = (routed + shared_contribution).reshape(B, L, hidden_dim)
    return result


def main():
    cfg = load_config()
    print("Loading all weights...")
    t0 = time.time()
    weights = load_all_weights()
    print(f"Weights loaded in {time.time()-t0:.1f}s")

    # Precompute RoPE
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    inv_freq = [1.0 / cfg["rope_theta"] ** (i / rotary_dim) for i in range(0, rotary_dim, 2)]
    inv_freq = torch.tensor(inv_freq, dtype=torch.float32)
    positions = torch.arange(cfg["max_position_embeddings"], dtype=torch.float32)
    freqs = positions.unsqueeze(1) @ inv_freq.unsqueeze(0)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    print(f"Input IDs: {input_ids}")
    x = F.embedding(torch.tensor([input_ids]), weights["model.embed_tokens.weight"].to(torch.bfloat16))
    print(f"Embed: mean={x.float().mean():.6f} min={x.float().min():.6f} max={x.float().max():.6f}")

    h = x
    kv_caches = {}

    for layer_idx in range(cfg["num_hidden_layers"]):
        # Input layernorm
        ln_w = weights[f"model.layers.{layer_idx}.input_layernorm.weight"].to(torch.bfloat16)
        h_normed = rms_norm(h, ln_w, cfg["rms_norm_eps"])

        # Token mixer
        is_full_attn = (layer_idx + 1) % cfg["full_attention_interval"] == 0
        if is_full_attn:
            attn_out, kv_caches[layer_idx] = attention_forward(
                h_normed, weights, f"model.layers.{layer_idx}.self_attn",
                cfg, cos, sin, kv_caches.get(layer_idx), 0)
        else:
            attn_out = deltanet_forward(
                h_normed, weights, f"model.layers.{layer_idx}.linear_attn", cfg, None)

        h = h + attn_out

        # Post-attn norm + MoE
        post_ln_w = weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].to(torch.bfloat16)
        h_normed2 = rms_norm(h, post_ln_w, cfg["rms_norm_eps"])
        moe_out = moe_forward(h_normed2, weights, f"model.layers.{layer_idx}.mlp", cfg)
        h = h + moe_out

        if layer_idx < 4 or layer_idx == 47:
            hf = h.float()
            print(f"Layer {layer_idx}: mean={hf.mean():.6f} min={hf.min():.6f} max={hf.max():.6f}")

    # Final norm
    final_ln_w = weights["model.norm.weight"].to(torch.bfloat16)
    h = rms_norm(h, final_ln_w, cfg["rms_norm_eps"])

    # LM head
    lm_w = weights["lm_head.weight"].to(torch.bfloat16)
    logits = F.linear(h[:, -1:], lm_w)
    logits_f = logits.squeeze().float()
    top5_ids = logits_f.argsort(descending=True)[:5]
    top5_vals = logits_f[top5_ids]
    print(f"\nTop-5 predictions:")
    for i in range(5):
        tid = top5_ids[i].item()
        val = top5_vals[i].item()
        token = tokenizer.decode([tid])
        print(f"  #{i+1}: id={tid} val={val:.4f} text={repr(token)}")


def load_config():
    with open(f"{MODEL_DIR}/config.json") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
