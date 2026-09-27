#!/usr/bin/env python
"""Compare next-token logits of Llama-2-7B and a KGW watermark student.

The script evaluates both models on *exactly the same tokenized contexts* and
computes ``delta_z(x) = logits_watermarked(x) - logits_base(x)``.  By default
one vector (the logits at the last non-padding token) is saved per prompt; this
is the vector used to predict the next token during ordinary causal inference.

Colab example::

    !pip install -q -U transformers accelerate huggingface_hub matplotlib
    !python analysis/compare_llama_logits.py --hf-token

The token is requested interactively with ``getpass`` when ``--hf-token`` is
not supplied.  It is never printed or written to the output directory.  A
Hugging Face token with access to ``meta-llama/Llama-2-7b-hf`` is required.
"""

import argparse
import csv
import gc
import getpass
import json
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

DEFAULT_BASE = "meta-llama/Llama-2-7b-hf"
DEFAULT_WATERMARKED = "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2"
DEFAULT_PROMPTS = [
    "Explain why the seasons change on Earth in a short paragraph.",
    "A careful scientist records uncertainty instead of hiding it.",
    "Write three practical suggestions for reducing household energy use.",
    "The old railway station stood at the edge of the town, where",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=DEFAULT_BASE, help="Clean Llama-2-7B model repo or local directory")
    p.add_argument("--watermarked", default=None, help="Watermarked model repo or local directory (defaults to cygu's k-specific checkpoint)")
    p.add_argument("--k", type=int, default=0, help="KGW k value used to select the default watermark repo")
    p.add_argument("--hf-token", nargs="?", const="__PROMPT__", default=None,
                   help="Use this token, or pass the flag without a value to enter it securely")
    p.add_argument("--prompt", dest="prompts", action="append", help="Input text x; repeat for multiple prompts")
    p.add_argument("--prompt-file", help="UTF-8 file with one input text per line")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--output-dir", default="analysis/llama_logit_diff_output")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--top-k", type=int, default=100, help="Number of largest positive/negative shifts to write to CSV")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_prompts(ns) -> List[str]:
    if ns.prompts:
        prompts = ns.prompts
    elif ns.prompt_file:
        with open(ns.prompt_file, encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("No input texts were supplied")
    return prompts


def get_token(ns):
    if ns.hf_token == "__PROMPT__":
        token = getpass.getpass("Hugging Face access token (input hidden): ")
    elif ns.hf_token:
        token = ns.hf_token
    else:
        token = os.environ.get("HF_TOKEN")
        if not token:
            token = getpass.getpass("Hugging Face access token (input hidden): ")
    if not token:
        raise ValueError("A Hugging Face token is required; it was empty")
    return token


def load_tokenizer(source, token, trust_remote_code):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source, token=token, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(source, token, device, trust_remote_code):
    from transformers import AutoModelForCausalLM
    kwargs = dict(token=token, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code)
    if device.type == "cuda":
        kwargs.update(torch_dtype=torch.float16, device_map="auto")
    else:
        kwargs.update(torch_dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if device.type == "cpu":
        model.to(device)
    model.eval()
    return model


def input_device(model):
    # With device_map="auto", input IDs must be placed on the first execution device.
    return model.get_input_embeddings().weight.device


def collect_next_logits(model, batches, device):
    vectors = []
    with torch.inference_mode():
        for encoded in batches:
            inputs = {k: v.to(device) for k, v in encoded.items()}
            out = model(**inputs, use_cache=False, return_dict=True)
            last = inputs["attention_mask"].sum(dim=1).long() - 1
            rows = torch.arange(last.shape[0], device=last.device)
            vectors.append(out.logits[rows, last].float().cpu())
            del out, inputs
    return torch.cat(vectors, dim=0).numpy()


def make_batches(tokenizer, prompts, max_length, batch_size):
    return [tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            for i in range(0, len(prompts), batch_size)]


def write_results(prompts, base_logits, wm_logits, tokenizer, output_dir, top_k, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    delta = wm_logits - base_logits
    np.savez_compressed(output_dir / "delta_logits.npz", delta=delta, base_logits=base_logits,
                        watermarked_logits=wm_logits)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    rows = []
    k = min(top_k, delta.shape[1])
    for i, prompt in enumerate(prompts):
        positive = np.argpartition(delta[i], -k)[-k:][::-1]
        negative = np.argpartition(delta[i], k - 1)[:k]
        for rank, token_id in enumerate(positive, 1):
            rows.append({"prompt_index": i, "prompt": prompt, "direction": "positive",
                         "rank": rank, "token_id": int(token_id),
                         "token": tokenizer.decode([int(token_id)]), "delta_logit": float(delta[i, token_id])})
        for rank, token_id in enumerate(negative, 1):
            rows.append({"prompt_index": i, "prompt": prompt, "direction": "negative",
                         "rank": rank, "token_id": int(token_id),
                         "token": tokenizer.decode([int(token_id)]), "delta_logit": float(delta[i, token_id])})
    with (output_dir / "top_logit_shifts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # A compact overview, while delta_logits.npz retains the full vocabulary vector.
    summary = [{"prompt_index": i, "mean_delta": float(delta[i].mean()),
                "rms_delta": float(np.sqrt(np.mean(delta[i] ** 2))),
                "max_positive": float(delta[i].max()), "min_negative": float(delta[i].min())}
               for i in range(len(prompts))]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return delta


def main():
    ns = parse_args()
    prompts = read_prompts(ns)
    if ns.watermarked is None:
        ns.watermarked = DEFAULT_WATERMARKED.replace("k0", f"k{ns.k}")
    token = get_token(ns)
    device = torch.device("cuda" if ns.device == "auto" and torch.cuda.is_available() else
                          "cpu" if ns.device == "auto" else ns.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"Device: {device}; prompts: {len(prompts)}; max length: {ns.max_length}")
    print(f"Base: {ns.base}\nWatermarked (k={ns.k}): {ns.watermarked}")

    base_tok = load_tokenizer(ns.base, token, ns.trust_remote_code)
    wm_tok = load_tokenizer(ns.watermarked, token, ns.trust_remote_code)
    base_batches = make_batches(base_tok, prompts, ns.max_length, ns.batch_size)
    wm_batches = make_batches(wm_tok, prompts, ns.max_length, ns.batch_size)
    for i, (a, b) in enumerate(zip(base_batches, wm_batches)):
        if not torch.equal(a["input_ids"], b["input_ids"]) or not torch.equal(a["attention_mask"], b["attention_mask"]):
            raise RuntimeError(f"Tokenizers differ in batch {i}; cannot compare logits under the same context")
    del wm_tok

    print("Loading clean model and collecting next-token logits...")
    base_model = load_model(ns.base, token, device, ns.trust_remote_code)
    base_logits = collect_next_logits(base_model, base_batches, input_device(base_model))
    del base_model; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    print("Loading watermarked model and collecting next-token logits...")
    wm_model = load_model(ns.watermarked, token, device, ns.trust_remote_code)
    wm_logits = collect_next_logits(wm_model, base_batches, input_device(wm_model))
    del wm_model; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    metadata = {"base": ns.base, "watermarked": ns.watermarked, "k": ns.k,
                "prompts": prompts, "definition": "delta = watermarked_logits - base_logits",
                "logit_position": "last non-padding token (next-token prediction)",
                "vocab_size": int(base_logits.shape[1]), "device": str(device)}
    delta = write_results(prompts, base_logits, wm_logits, base_tok, Path(ns.output_dir), ns.top_k, metadata)
    print(f"Saved full delta vectors with shape {delta.shape} to {ns.output_dir}/delta_logits.npz")
    print(f"Mean RMS shift: {np.sqrt(np.mean(delta ** 2), axis=1).mean():.6f}")


if __name__ == "__main__":
    main()
