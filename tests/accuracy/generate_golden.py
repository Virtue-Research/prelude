#!/usr/bin/env python3
"""Generate golden reference outputs using HuggingFace transformers.

Usage:
    python tests/accuracy/generate_golden.py --model Qwen/Qwen3-0.6B --dtype float32
    python tests/accuracy/generate_golden.py --model Qwen/Qwen3-0.6B --dtype bfloat16
"""
import argparse
import json
import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_chatml_prompt(messages: list[dict]) -> str:
    """Build ChatML prompt matching rust-infer server's chat_to_engine_request().

    Server code (main.rs:564-571):
        messages.iter()
            .map(|m| format!("<|im_start|>{}\\n{}<|im_end|>", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\\n")
            + "\\n<|im_start|>assistant\\n"
    """
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def generate_for_prompt(model, tokenizer, prompt_spec, device):
    """Generate output for a single prompt spec."""
    if prompt_spec.get("type") == "completion":
        prompt_text = prompt_spec["prompt"]
    else:
        prompt_text = build_chatml_prompt(prompt_spec["messages"])

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=prompt_spec["max_tokens"],
            do_sample=False,
        )

    output_ids = outputs[0, prompt_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)

    # Determine finish reason by checking for EOS tokens
    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eid = tokenizer.eos_token_id
        if isinstance(eid, list):
            eos_ids.update(eid)
        else:
            eos_ids.add(eid)
    if hasattr(model, "generation_config") and model.generation_config.eos_token_id is not None:
        eid = model.generation_config.eos_token_id
        if isinstance(eid, list):
            eos_ids.update(eid)
        else:
            eos_ids.add(eid)

    finish_reason = "length"
    if output_ids and output_ids[-1] in eos_ids:
        finish_reason = "stop"

    # Also check what HuggingFace's chat template produces (diagnostic only)
    hf_prompt = None
    if prompt_spec.get("type") != "completion":
        try:
            hf_prompt = tokenizer.apply_chat_template(
                prompt_spec["messages"], tokenize=False, add_generation_prompt=True
            )
            if hf_prompt != prompt_text:
                print(f"  WARNING: ChatML template mismatch for {prompt_spec['id']}!")
                print(f"    Server template: {prompt_text[:100]}...")
                print(f"    HF template:     {hf_prompt[:100]}...")
        except Exception:
            pass

    return {
        "output_token_ids": output_ids,
        "output_text": output_text,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_len,
        "completion_tokens": len(output_ids),
        "prompt_text": prompt_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate golden reference outputs")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--prompts", default=None, help="Path to prompts.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for golden JSON")
    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    prompts_path = Path(args.prompts) if args.prompts else script_dir / "golden" / "prompts.json"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "golden"

    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = "cpu"

    print(f"Loading {args.model} in {args.dtype} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    with open(prompts_path) as f:
        prompts_data = json.load(f)

    results = {}
    for spec in prompts_data["prompts"]:
        pid = spec["id"]
        print(f"  Generating: {pid} (max_tokens={spec['max_tokens']})...")
        results[pid] = generate_for_prompt(model, tokenizer, spec, device)
        print(f"    -> {results[pid]['completion_tokens']} tokens, "
              f"finish_reason={results[pid]['finish_reason']}")

    model_slug = args.model.split("/")[-1].lower()
    dtype_slug = "f32" if args.dtype == "float32" else "bf16"
    out_path = output_dir / f"{model_slug}-{dtype_slug}.json"

    golden = {
        "model": args.model,
        "dtype": args.dtype,
        "generator": "transformers",
        "generator_version": transformers.__version__,
        "torch_version": torch.__version__,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
