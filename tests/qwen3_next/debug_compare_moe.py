"""Compare Layer 0 MoE output between HF model and our manual implementation."""
import torch
import torch.nn.functional as F
import json
import time
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/path/to/qwen3-next"


def load_raw_weights(keys_needed):
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        index = json.load(f)
    weights = {}
    shards_needed = set()
    for key in keys_needed:
        if key in index["weight_map"]:
            shards_needed.add(index["weight_map"][key])
    for shard in shards_needed:
        with safe_open(f"{MODEL_DIR}/{shard}", framework="pt") as f:
            for key in f.keys():
                if key in keys_needed:
                    weights[key] = f.get_tensor(key)
    return weights


def main():
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    layer0 = model.model.layers[0]
    moe = layer0.mlp

    # Use register_forward_hook to capture MoE input/output
    captured = {}

    def hook_moe_input(module, args, kwargs):
        captured["moe_input"] = args[0].detach().clone()

    def hook_moe_output(module, args, kwargs, output):
        if isinstance(output, tuple):
            captured["moe_output"] = output[0].detach().clone()
        else:
            captured["moe_output"] = output.detach().clone()

    def hook_deltanet_output(module, args, kwargs, output):
        captured["deltanet_output"] = output.detach().clone()

    moe.register_forward_pre_hook(hook_moe_input, with_kwargs=True)
    moe.register_forward_hook(hook_moe_output, with_kwargs=True)
    layer0.linear_attn.register_forward_hook(hook_deltanet_output, with_kwargs=True)

    # Run forward
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    input_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    print(f"Input IDs: {input_ids}")

    with torch.no_grad():
        output = model(torch.tensor([input_ids]))

    logits = output.logits[0, -1].float()
    top_id = logits.argmax().item()
    print(f"\nHF result: id={top_id} val={logits[top_id]:.4f} text={repr(tokenizer.decode([top_id]))}")

    # Print captured values
    print(f"\nDeltaNet output: mean={captured['deltanet_output'].float().mean():.6f} "
          f"min={captured['deltanet_output'].float().min():.6f} max={captured['deltanet_output'].float().max():.6f}")
    print(f"MoE input: mean={captured['moe_input'].float().mean():.6f} "
          f"min={captured['moe_input'].float().min():.6f} max={captured['moe_input'].float().max():.6f}")
    print(f"MoE output: mean={captured['moe_output'].float().mean():.6f} "
          f"min={captured['moe_output'].float().min():.6f} max={captured['moe_output'].float().max():.6f}")

    # Now run manual MoE on the same input
    print("\n" + "=" * 60)
    print("Manual MoE comparison")
    print("=" * 60)

    moe_input = captured["moe_input"]
    B, L, hidden_dim = moe_input.shape
    xs = moe_input.reshape(-1, hidden_dim)

    cfg = json.load(open(f"{MODEL_DIR}/config.json"))
    prefix = "model.layers.0.mlp"

    # First check: HF expert weight vs raw safetensors
    print("\n--- Expert weight fusing verification ---")
    expert0_keys = [
        f"{prefix}.experts.0.gate_proj.weight",
        f"{prefix}.experts.0.up_proj.weight",
        f"{prefix}.experts.0.down_proj.weight",
    ]
    raw0 = load_raw_weights(set(expert0_keys))

    hf_gate_up_0 = moe.experts.gate_up_proj[0].data  # [1024, 2048]
    raw_gate_0 = raw0[f"{prefix}.experts.0.gate_proj.weight"]  # [512, 2048]
    raw_up_0 = raw0[f"{prefix}.experts.0.up_proj.weight"]  # [512, 2048]

    fused_gate_up = torch.cat([raw_gate_0, raw_up_0], dim=0)
    fused_up_gate = torch.cat([raw_up_0, raw_gate_0], dim=0)
    print(f"  gate_up = cat(gate, up): match={torch.equal(hf_gate_up_0, fused_gate_up)}")
    print(f"  gate_up = cat(up, gate): match={torch.equal(hf_gate_up_0, fused_up_gate)}")

    hf_down_0 = moe.experts.down_proj[0].data  # [2048, 512]
    raw_down_0 = raw0[f"{prefix}.experts.0.down_proj.weight"]  # [2048, 512]
    print(f"  down_proj matches: {torch.equal(hf_down_0, raw_down_0)}")

    # Check router weight
    print("\n--- Router weight ---")
    raw_gate_keys = [f"{prefix}.gate.weight"]
    raw_gate = load_raw_weights(set(raw_gate_keys))
    hf_router_w = moe.gate.weight.data  # [512, 2048]
    raw_router_w = raw_gate.get(f"{prefix}.gate.weight")
    if raw_router_w is not None:
        print(f"  Router weight match: {torch.equal(hf_router_w, raw_router_w)}")
    else:
        print("  Router weight not found in safetensors!")

    # Check shared expert gate weight
    print("\n--- Shared expert gate ---")
    seg_keys = [f"{prefix}.shared_expert_gate.weight"]
    raw_seg = load_raw_weights(set(seg_keys))
    hf_seg_w = moe.shared_expert_gate.weight.data  # [1, 2048]
    raw_seg_w = raw_seg.get(f"{prefix}.shared_expert_gate.weight")
    if raw_seg_w is not None:
        print(f"  Shape: HF={list(hf_seg_w.shape)}, raw={list(raw_seg_w.shape)}")
        if hf_seg_w.shape == raw_seg_w.shape:
            print(f"  Match: {torch.equal(hf_seg_w, raw_seg_w)}")
        else:
            # Maybe raw is [2048] and HF is [1, 2048]
            if raw_seg_w.shape == (2048,):
                print(f"  Match (after unsqueeze): {torch.equal(hf_seg_w, raw_seg_w.unsqueeze(0))}")

    # Now do manual MoE computation
    print("\n--- Manual MoE computation ---")

    # Load all needed weights
    all_moe_keys = set()
    all_moe_keys.add(f"{prefix}.gate.weight")
    all_moe_keys.add(f"{prefix}.shared_expert.gate_proj.weight")
    all_moe_keys.add(f"{prefix}.shared_expert.up_proj.weight")
    all_moe_keys.add(f"{prefix}.shared_expert.down_proj.weight")
    all_moe_keys.add(f"{prefix}.shared_expert_gate.weight")

    # Figure out which experts are needed (run router first)
    gate_w = raw_gate[f"{prefix}.gate.weight"].to(torch.bfloat16)
    router_logits = F.linear(xs.float(), gate_w.float())
    routing_weights = F.softmax(router_logits, dim=-1)
    topk_vals, topk_ids = torch.topk(routing_weights, cfg["num_experts_per_tok"], dim=-1)
    print(f"  topk_ids: {topk_ids.numpy()}")
    print(f"  topk_vals sum per token: {topk_vals.sum(-1).numpy()}")

    if cfg.get("norm_topk_prob", False):
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        print(f"  topk_vals (normalized) sum: {topk_vals.sum(-1).numpy()}")

    # Get unique expert IDs
    unique_experts = sorted(set(topk_ids.flatten().numpy().tolist()))
    print(f"  Unique experts needed: {unique_experts}")

    for eid in unique_experts:
        all_moe_keys.add(f"{prefix}.experts.{eid}.gate_proj.weight")
        all_moe_keys.add(f"{prefix}.experts.{eid}.up_proj.weight")
        all_moe_keys.add(f"{prefix}.experts.{eid}.down_proj.weight")

    raw = load_raw_weights(all_moe_keys)

    # Shared expert
    se_gw = raw[f"{prefix}.shared_expert.gate_proj.weight"].to(torch.bfloat16)
    se_uw = raw[f"{prefix}.shared_expert.up_proj.weight"].to(torch.bfloat16)
    se_dw = raw[f"{prefix}.shared_expert.down_proj.weight"].to(torch.bfloat16)
    shared_gate = F.silu(F.linear(xs, se_gw))
    shared_up = F.linear(xs, se_uw)
    shared_out = F.linear(shared_gate * shared_up, se_dw)

    seg_w = raw[f"{prefix}.shared_expert_gate.weight"].to(torch.bfloat16)
    # Handle shape: raw might be [2048] or [1, 2048]
    if seg_w.dim() == 1:
        seg_w = seg_w.unsqueeze(0)
    shared_gate_val = torch.sigmoid(F.linear(xs, seg_w))
    shared_contribution = shared_out * shared_gate_val

    print(f"  Shared expert out: mean={shared_out.float().mean():.6f}")
    print(f"  Shared gate sigmoid: mean={shared_gate_val.float().mean():.6f} "
          f"val={shared_gate_val[0,0].item():.6f}")
    print(f"  Shared contribution: mean={shared_contribution.float().mean():.6f}")

    # Compare shared expert with HF
    with torch.no_grad():
        hf_shared = moe.shared_expert(xs)
        hf_seg = torch.sigmoid(moe.shared_expert_gate(xs))
        hf_shared_contrib = hf_shared * hf_seg

    diff_shared = (shared_contribution.float() - hf_shared_contrib.float()).abs().max().item()
    print(f"  Shared contribution diff vs HF: {diff_shared:.6f}")
    print(f"  HF shared gate: {hf_seg[0,0].item():.6f}")

    # Routed experts
    routed = torch.zeros_like(xs)
    topk_ids_np = topk_ids.cpu().numpy()
    topk_vals_np = topk_vals.cpu().float().numpy()

    for expert_id in unique_experts:
        mask = (topk_ids_np == expert_id)
        token_positions = mask.any(axis=1).nonzero()[0]
        if len(token_positions) == 0:
            continue

        e_gw = raw[f"{prefix}.experts.{expert_id}.gate_proj.weight"].to(torch.bfloat16)
        e_uw = raw[f"{prefix}.experts.{expert_id}.up_proj.weight"].to(torch.bfloat16)
        e_dw = raw[f"{prefix}.experts.{expert_id}.down_proj.weight"].to(torch.bfloat16)

        x_sub = xs[token_positions]
        e_gate = F.silu(F.linear(x_sub, e_gw))
        e_up = F.linear(x_sub, e_uw)
        e_out = F.linear(e_gate * e_up, e_dw)

        for j, tid in enumerate(token_positions):
            w = topk_vals_np[tid][mask[tid]].sum()
            routed[tid] += (e_out[j] * w).to(routed.dtype)

    manual_moe_out = (routed + shared_contribution).reshape(B, L, hidden_dim)

    # Compare with HF
    hf_moe_out = captured["moe_output"]
    print(f"\n  Manual MoE out: mean={manual_moe_out.float().mean():.6f} "
          f"min={manual_moe_out.float().min():.6f} max={manual_moe_out.float().max():.6f}")
    print(f"  HF MoE out:     mean={hf_moe_out.float().mean():.6f} "
          f"min={hf_moe_out.float().min():.6f} max={hf_moe_out.float().max():.6f}")

    diff = (manual_moe_out.float() - hf_moe_out.float()).abs()
    print(f"  Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")

    if diff.max().item() > 0.01:
        print("\n  *** MoE OUTPUT DIFFERS! ***")

        # Debug: run HF experts directly on the same input
        with torch.no_grad():
            _, hf_routing_weights, hf_selected = moe.gate(xs)
            hf_expert_out = moe.experts(xs, hf_selected, hf_routing_weights)

        print(f"\n  HF routing indices: {hf_selected.numpy()}")
        print(f"  Our routing indices: {topk_ids.numpy()}")
        print(f"  Routing match: {torch.equal(hf_selected, topk_ids)}")

        print(f"\n  HF expert output: mean={hf_expert_out.float().mean():.6f} "
              f"min={hf_expert_out.float().min():.6f} max={hf_expert_out.float().max():.6f}")
        print(f"  Our routed output: mean={routed.float().mean():.6f} "
              f"min={routed.float().min():.6f} max={routed.float().max():.6f}")

        diff_expert = (routed.float() - hf_expert_out.float()).abs()
        print(f"  Expert output diff: max={diff_expert.max().item():.6f}, mean={diff_expert.mean().item():.6f}")

        # Test single expert to find exact difference
        test_eid = unique_experts[0]
        print(f"\n  --- Testing single expert {test_eid} ---")

        # HF fused expert
        hf_gate_up_w = moe.experts.gate_up_proj[test_eid].data
        hf_down_w = moe.experts.down_proj[test_eid].data
        test_input = xs[0:1]

        hf_gate_up_out = F.linear(test_input, hf_gate_up_w)
        hf_gate, hf_up = hf_gate_up_out.chunk(2, dim=-1)
        hf_expert_single = F.linear(F.silu(hf_gate) * hf_up, hf_down_w)

        # Our separate expert
        our_gw = raw[f"{prefix}.experts.{test_eid}.gate_proj.weight"].to(torch.bfloat16)
        our_uw = raw[f"{prefix}.experts.{test_eid}.up_proj.weight"].to(torch.bfloat16)
        our_dw = raw[f"{prefix}.experts.{test_eid}.down_proj.weight"].to(torch.bfloat16)
        our_gate = F.silu(F.linear(test_input, our_gw))
        our_up = F.linear(test_input, our_uw)
        our_expert_single = F.linear(our_gate * our_up, our_dw)

        diff_single = (hf_expert_single.float() - our_expert_single.float()).abs()
        print(f"  HF single expert: mean={hf_expert_single.float().mean():.6f}")
        print(f"  Our single expert: mean={our_expert_single.float().mean():.6f}")
        print(f"  Diff: max={diff_single.max().item():.6f}")
    else:
        print("\n  >>> MoE OUTPUT MATCHES! <<<")


if __name__ == "__main__":
    main()
