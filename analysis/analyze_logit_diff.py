#!/usr/bin/env python
"""Export same-context logit differences between two causal language models.

This script intentionally performs no plots or statistical analysis. It only
loads two models, computes next-token logits for identical token IDs, and
saves NumPy arrays under ``analysis/result/``.

Colab example::

    !python analysis/analyze_logit_diff.py --hf-token

    # Compare arbitrary local directories or Hugging Face repositories:
    !python analysis/analyze_logit_diff.py \
        --base meta-llama/Llama-2-7b-hf \
        --watermarked cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2 \
        --hf-token
"""

import argparse
import gc
import getpass
import json
import os
from pathlib import Path

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
    p.add_argument("--base", default=DEFAULT_BASE, help="Clean/base model repository or local directory")
    p.add_argument("--watermarked", default=DEFAULT_WATERMARKED, help="Second model repository or local directory")
    p.add_argument("--prompt", dest="prompts", action="append", help="Input text; repeat for multiple prompts")
    p.add_argument("--prompt-file", help="UTF-8 file containing one input text per line")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--output-dir", default="analysis/result")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--hf-token", nargs="?", const="__PROMPT__", default=None,
                   help="Token value, or pass without value to enter it securely")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_prompts(ns):
    if ns.prompts:
        prompts = ns.prompts
    elif ns.prompt_file:
        with open(ns.prompt_file, encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("No prompts were supplied")
    return prompts


def get_token(value):
    if value == "__PROMPT__":
        return getpass.getpass("Hugging Face access token (input hidden): ")
    token = value or os.environ.get("HF_TOKEN")
    if not token:
        token = getpass.getpass("Hugging Face access token (input hidden): ")
    if not token:
        raise ValueError("A Hugging Face token is required")
    return token


def load_tokenizer(source, token, trust_remote_code):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        source, token=token, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(source, token, device, trust_remote_code):
    from transformers import AutoModelForCausalLM
    kwargs = {"token": token, "low_cpu_mem_usage": True,
              "trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        kwargs.update(torch_dtype=torch.float16, device_map="auto")
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if device.type == "cpu":
        model.to(device)
    model.eval()
    return model


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def tokenize(tokenizer, prompts, max_length, batch_size):
    return [tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            for i in range(0, len(prompts), batch_size)]


def check_same_inputs(base_batches, other_batches):
    if len(base_batches) != len(other_batches):
        raise RuntimeError("The two tokenizers produced different batch counts")
    for i, (base, other) in enumerate(zip(base_batches, other_batches)):
        for key in ("input_ids", "attention_mask"):
            if not torch.equal(base[key], other[key]):
                raise RuntimeError(f"Tokenizers differ in batch {i} ({key})")


def next_token_logits(model, batches):
    device = model_input_device(model)
    outputs = []
    with torch.inference_mode():
        for encoded in batches:
            inputs = {key: value.to(device) for key, value in encoded.items()}
            result = model(**inputs, use_cache=False, return_dict=True)
            last = inputs["attention_mask"].sum(dim=1).long() - 1
            rows = torch.arange(last.shape[0], device=device)
            outputs.append(result.logits[rows, last].float().cpu())
    return torch.cat(outputs).numpy()


def clear_model(model, device):
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ns = parse_args()
    prompts = read_prompts(ns)
    token = get_token(ns.hf_token)
    device = torch.device("cuda" if ns.device == "auto" and torch.cuda.is_available()
                          else "cpu" if ns.device == "auto" else ns.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    base_tokenizer = load_tokenizer(ns.base, token, ns.trust_remote_code)
    other_tokenizer = load_tokenizer(ns.watermarked, token, ns.trust_remote_code)
    base_batches = tokenize(base_tokenizer, prompts, ns.max_length, ns.batch_size)
    other_batches = tokenize(other_tokenizer, prompts, ns.max_length, ns.batch_size)
    check_same_inputs(base_batches, other_batches)

    print(f"Device: {device}; prompts: {len(prompts)}")
    print(f"Base: {ns.base}")
    print(f"Other: {ns.watermarked}")

    print("Collecting base logits...")
    base_model = load_model(ns.base, token, device, ns.trust_remote_code)
    base_logits = next_token_logits(base_model, base_batches)
    clear_model(base_model, device)

    print("Collecting other-model logits...")
    other_model = load_model(ns.watermarked, token, device, ns.trust_remote_code)
    other_logits = next_token_logits(other_model, base_batches)
    clear_model(other_model, device)

    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta = other_logits - base_logits
    np.save(output_dir / "base_logits.npy", base_logits)
    np.save(output_dir / "other_logits.npy", other_logits)
    np.save(output_dir / "delta_logits.npy", delta)
    metadata = {
        "base": ns.base,
        "other": ns.watermarked,
        "prompts": prompts,
        "max_length": ns.max_length,
        "definition": "delta_logits = other_logits - base_logits",
        "position": "last non-padding token; next-token logits",
        "shape": list(delta.shape),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved NumPy data to {output_dir.resolve()}")
    print(f"delta_logits shape: {delta.shape}")


if __name__ == "__main__":
    main()
