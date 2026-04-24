#!/usr/bin/env python3
"""Reference script for Gemma 4 E2B-it forward pass.
Runs a single prefill pass and prints intermediate values for debugging.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "The meaning of life is"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

def main():
    print(f"Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model from {MODEL_ID} (dtype={DTYPE}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    print(f"\n{'='*60}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Input token IDs: {input_ids[0].tolist()}")
    print(f"Tokens: {[tokenizer.decode(t) for t in input_ids[0]]}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    logits = out.logits  # (1, seq_len, vocab_size)
    hidden_states = out.hidden_states  # tuple of (1, seq_len, hidden_dim)

    # --- Hidden states after embedding (layer 0 = embed output) ---
    emb = hidden_states[0][0, -1]  # last token
    print("Hidden states after embedding (last token):")
    print(f"  first 5: {emb[:5].float().tolist()}")
    print(f"  mean:    {emb.float().mean().item():.6f}")
    print()

    # --- Hidden states after each transformer layer ---
    for i, hs in enumerate(hidden_states[1:], start=1):
        h = hs[0, -1]
        print(f"Layer {i:3d} hidden (last token):  "
              f"first5={[f'{v:.4f}' for v in h[:5].float().tolist()]}  "
              f"mean={h.float().mean().item():.6f}")
    print()

    # --- Logits for the last token (top-5) ---
    last_logits = logits[0, -1]  # (vocab_size,)
    probs = torch.softmax(last_logits.float(), dim=-1)
    top5 = torch.topk(probs, 5)

    print("Top-5 predictions for next token:")
    for i in range(5):
        tid = top5.indices[i].item()
        p = top5.values[i].item()
        tok = tokenizer.decode(tid)
        print(f"  {tid:>8d}  {p:.4%}  {tok!r}")

    # --- Predicted next token ---
    pred_id = last_logits.argmax().item()
    pred_tok = tokenizer.decode(pred_id)
    print(f"\nPredicted next token: id={pred_id}  text={pred_tok!r}")

if __name__ == "__main__":
    main()
