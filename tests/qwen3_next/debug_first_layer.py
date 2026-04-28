"""Debug: compare first DeltaNet layer output against manual Python computation."""
import torch
import torch.nn.functional as F
import json
import sys
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = "/path/to/qwen3-next"

def load_config():
    with open(f"{MODEL_DIR}/config.json") as f:
        return json.load(f)

def load_weights(prefix, weight_names):
    """Load specific weights from safetensors shards."""
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        index = json.load(f)

    weights = {}
    shard_files = set()
    for name in weight_names:
        full_name = f"{prefix}.{name}" if prefix else name
        if full_name in index["weight_map"]:
            shard_files.add(index["weight_map"][full_name])

    for shard_file in shard_files:
        path = f"{MODEL_DIR}/{shard_file}"
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix) or key in weight_names:
                    weights[key] = f.get_tensor(key)
    return weights

def manual_rms_norm(x, weight, eps=1e-6):
    """RMSNorm on last dimension."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normed = x_f32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(x.dtype)

def l2_normalize(x, eps=1e-6):
    """L2 normalize along last dimension."""
    norm = torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)
    return x / norm

def softplus(x, beta=1.0, threshold=20.0):
    """Numerically stable softplus."""
    mask = (beta * x) <= threshold
    safe = torch.where(mask, x, torch.zeros_like(x))
    sp = (1.0 / beta) * torch.log1p(torch.exp(beta * safe))
    return torch.where(mask, sp, x)

def main():
    cfg = load_config()

    hidden_size = cfg["hidden_size"]  # 2048
    num_k_heads = cfg["linear_num_key_heads"]  # 16
    num_v_heads = cfg["linear_num_value_heads"]  # 32
    head_k_dim = cfg["linear_key_head_dim"]  # 128
    head_v_dim = cfg["linear_value_head_dim"]  # 128
    conv_kernel = cfg["linear_conv_kernel_dim"]  # 4

    key_dim = num_k_heads * head_k_dim  # 2048
    value_dim = num_v_heads * head_v_dim  # 4096
    conv_dim = key_dim * 2 + value_dim  # 8192
    kv_ratio = num_v_heads // num_k_heads  # 2

    print(f"Config: key_dim={key_dim} value_dim={value_dim} conv_dim={conv_dim} kv_ratio={kv_ratio}")

    # Load embedding + layer 0 weights
    weight_names = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.linear_attn.in_proj_qkvz.weight",
        "model.layers.0.linear_attn.in_proj_ba.weight",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.linear_attn.out_proj.weight",
    ]

    print("Loading weights...")
    weights = {}
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        index = json.load(f)

    shard_files = set()
    for name in weight_names:
        if name in index["weight_map"]:
            shard_files.add(index["weight_map"][name])
        else:
            print(f"WARNING: {name} not found in index!")

    for shard_file in shard_files:
        path = f"{MODEL_DIR}/{shard_file}"
        print(f"  Loading {shard_file}...")
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key in weight_names:
                    weights[key] = f.get_tensor(key)

    for name in weight_names:
        if name in weights:
            print(f"  {name}: {weights[name].shape} {weights[name].dtype}")
        else:
            print(f"  {name}: MISSING!")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    print(f"\nInput IDs: {input_ids}")
    input_tensor = torch.tensor([input_ids])

    # Embedding
    embed_weight = weights["model.embed_tokens.weight"].to(torch.bfloat16)
    x = F.embedding(input_tensor, embed_weight)  # [1, L, 2048]
    print(f"Embedding: shape={x.shape} mean={x.float().mean():.6f} min={x.float().min():.6f} max={x.float().max():.6f}")

    # Input LayerNorm
    ln_weight = weights["model.layers.0.input_layernorm.weight"].to(torch.bfloat16)
    h = manual_rms_norm(x, ln_weight, eps=cfg["rms_norm_eps"])
    print(f"After LN: mean={h.float().mean():.6f} min={h.float().min():.6f} max={h.float().max():.6f}")

    # in_proj_qkvz: [proj_dim, hidden_size]
    qkvz_w = weights["model.layers.0.linear_attn.in_proj_qkvz.weight"].to(torch.bfloat16)
    full_proj = F.linear(h, qkvz_w)  # [1, L, proj_dim]
    print(f"After QKVZ proj: shape={full_proj.shape} mean={full_proj.float().mean():.6f}")

    # in_proj_ba
    ba_w = weights["model.layers.0.linear_attn.in_proj_ba.weight"].to(torch.bfloat16)
    ba = F.linear(h, ba_w)  # [1, L, num_v_heads*2]
    print(f"After BA proj: shape={ba.shape}")

    # Interleaved QKVZ split (SGLang style)
    B, L, _ = full_proj.shape
    group_dim = head_k_dim * 2 + kv_ratio * head_v_dim * 2  # 768
    full_proj_g = full_proj.view(B, L, num_k_heads, group_dim)

    q = full_proj_g[..., :head_k_dim]  # [1, L, 16, 128]
    k = full_proj_g[..., head_k_dim:2*head_k_dim]
    v_offset = 2 * head_k_dim
    v = full_proj_g[..., v_offset:v_offset + kv_ratio*head_v_dim]  # [1, L, 16, 256]
    z = full_proj_g[..., v_offset + kv_ratio*head_v_dim:]

    q = q.reshape(B, L, key_dim)
    k = k.reshape(B, L, key_dim)
    v = v.reshape(B, L, value_dim)
    z = z.reshape(B, L, value_dim)

    print(f"Q: mean={q.float().mean():.6f} K: mean={k.float().mean():.6f} V: mean={v.float().mean():.6f} Z: mean={z.float().mean():.6f}")

    # BA split
    ba_group = ba.view(B, L, num_k_heads, kv_ratio * 2)
    b_param = ba_group[..., :kv_ratio].reshape(B, L, num_v_heads)
    a_param = ba_group[..., kv_ratio:].reshape(B, L, num_v_heads)
    print(f"b_param: mean={b_param.float().mean():.6f} a_param: mean={a_param.float().mean():.6f}")

    # Conv1d
    conv_w = weights["model.layers.0.linear_attn.conv1d.weight"].to(torch.bfloat16)  # [conv_dim, 1, kernel]
    print(f"Conv weight: shape={conv_w.shape}")

    qkv = torch.cat([q, k, v], dim=-1)  # [1, L, conv_dim]
    qkv_t = qkv.transpose(1, 2)  # [1, conv_dim, L]

    # Pad with zeros for causal conv
    pad = torch.zeros(1, conv_dim, conv_kernel - 1, dtype=torch.bfloat16)
    padded = torch.cat([pad, qkv_t], dim=2)

    # Depthwise conv1d
    conv_w_f32 = conv_w.float()
    padded_f32 = padded.float()
    conv_out = F.conv1d(padded_f32, conv_w_f32, padding=0, stride=1, dilation=1, groups=conv_dim)
    conv_out = conv_out.to(torch.bfloat16).transpose(1, 2)  # [1, L, conv_dim]

    # SiLU activation
    conv_out = F.silu(conv_out)

    # Split into q, k, v
    q_conv = conv_out[..., :key_dim]
    k_conv = conv_out[..., key_dim:2*key_dim]
    v_conv = conv_out[..., 2*key_dim:]

    print(f"After conv+silu: Q mean={q_conv.float().mean():.6f} K mean={k_conv.float().mean():.6f} V mean={v_conv.float().mean():.6f}")

    # Gating params
    dt_bias = weights["model.layers.0.linear_attn.dt_bias"]
    A_log = weights["model.layers.0.linear_attn.A_log"]
    print(f"dt_bias: {dt_bias[:5]} A_log: {A_log[:5]}")

    # Delta rule recurrence
    norm_w = weights["model.layers.0.linear_attn.norm.weight"].to(torch.bfloat16)
    out_proj_w = weights["model.layers.0.linear_attn.out_proj.weight"].to(torch.bfloat16)

    state = torch.zeros(num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
    outputs = []

    scale = head_k_dim ** -0.5

    for t in range(L):
        q_t = q_conv[0, t]  # [key_dim]
        k_t = k_conv[0, t]
        v_t = v_conv[0, t]  # [value_dim]
        b_t = b_param[0, t]  # [num_v_heads]
        a_t = a_param[0, t]

        # Reshape to heads
        q_h = q_t.reshape(num_k_heads, head_k_dim).float()
        k_h = k_t.reshape(num_k_heads, head_k_dim).float()
        v_h = v_t.reshape(num_v_heads, head_v_dim).float()

        # L2 normalize
        q_h = l2_normalize(q_h)
        k_h = l2_normalize(k_h)

        # Expand k from [K_heads, k_dim] to [V_heads, k_dim]
        k_exp = k_h.repeat_interleave(kv_ratio, dim=0)  # [32, 128]
        q_exp = q_h.repeat_interleave(kv_ratio, dim=0)  # [32, 128]

        # Scale q
        q_exp = q_exp * scale

        # Compute gating
        g = -A_log.float().exp() * softplus(a_t.float() + dt_bias.float())  # [32]
        beta = b_t.float().sigmoid()  # [32]

        # 1. Decay state
        state = state * g.exp().unsqueeze(1).unsqueeze(2)

        # 2. Delta correction: v -= state^T @ k
        prediction = torch.bmm(state.transpose(1, 2), k_exp.unsqueeze(2)).squeeze(2)  # [32, v_dim]
        v_residual = v_h - prediction

        # 3. Gate residual
        v_gated = v_residual * beta.unsqueeze(1)

        # 4. Update state: state += outer(k, v_gated)
        state = state + torch.bmm(k_exp.unsqueeze(2), v_gated.unsqueeze(1))

        # 5. Output = state^T @ q
        output = torch.bmm(state.transpose(1, 2), q_exp.unsqueeze(2)).squeeze(2)  # [32, v_dim]
        outputs.append(output)

    output_stack = torch.stack(outputs, dim=0).to(torch.bfloat16)  # [L, V_heads, v_dim]
    output_flat = output_stack.reshape(L, value_dim)  # [L, value_dim]
    print(f"DeltaNet output: shape={output_flat.shape} mean={output_flat.float().mean():.6f} min={output_flat.float().min():.6f} max={output_flat.float().max():.6f}")

    # Gated RMSNorm: norm(output) * silu(z)
    z_2d = z.squeeze(0)  # [L, value_dim]
    output_for_norm = output_flat.reshape(L, num_v_heads, head_v_dim)
    z_for_norm = z_2d.reshape(L, num_v_heads, head_v_dim)

    normed = manual_rms_norm(output_for_norm, norm_w, eps=cfg["rms_norm_eps"])
    gated = normed * F.silu(z_for_norm)
    gated_flat = gated.reshape(L, value_dim)  # [L, value_dim]
    print(f"After GatedNorm: mean={gated_flat.float().mean():.6f} min={gated_flat.float().min():.6f} max={gated_flat.float().max():.6f}")

    # Output projection
    attn_out = F.linear(gated_flat, out_proj_w)  # [L, hidden_size]
    print(f"DeltaNet final: mean={attn_out.float().mean():.6f} min={attn_out.float().min():.6f} max={attn_out.float().max():.6f}")

    # Residual connection
    layer_out_attn = x.squeeze(0) + attn_out  # [L, hidden_size]
    print(f"After attn residual: mean={layer_out_attn.float().mean():.6f} min={layer_out_attn.float().min():.6f} max={layer_out_attn.float().max():.6f}")

    print("\nPer-token last-dim first 5 values of DeltaNet output:")
    for t in range(min(L, 3)):
        vals = output_flat[t, :5].float().tolist()
        print(f"  t={t}: {vals}")

if __name__ == "__main__":
    main()
