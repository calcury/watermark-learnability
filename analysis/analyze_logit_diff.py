#!/usr/bin/env python
"""Export next-token logits for a chosen model-family / variant pair.

Choose a model family (``llama`` or ``pythia``) and two variants
(``base``, ``k0``, ``k1``, ``k2``). Model IDs and local cache paths mirror
``analysis/fetch_llama.py`` and ``analysis/fetch_pythia.py``.

Examples::

    !python analysis/analyze_logit_diff.py llama base k0
    !python analysis/analyze_logit_diff.py pythia base k2

For gated Llama access, set ``HF_TOKEN`` or enter it when prompted.
"""

import argparse
import getpass
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

MODEL_REPOS = {
    "llama": {
        "base": "meta-llama/Llama-2-7b-hf",
        "k0": "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2",
        "k1": "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2",
        "k2": "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2",
    },
    "pythia": {
        "base": "EleutherAI/pythia-1.4b",
        "k0": "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2",
        "k1": "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2",
        "k2": "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k2-gamma0.25-delta2",
    },
}
DEFAULT_PROMPTS = [
    "Explain why the seasons change on Earth in a short paragraph.",
    "A careful scientist records uncertainty instead of hiding it.",
    "Write three practical suggestions for reducing household energy use.",
    "The old railway station stood at the edge of the town, where",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=MODEL_REPOS, help="Model family: llama or pythia")
    parser.add_argument("model_a", choices=("base", "k0", "k1", "k2"))
    parser.add_argument("model_b", choices=("base", "k0", "k1", "k2"))
    parser.add_argument("--prompt", dest="prompts", action="append", help="Input text; repeat to compare multiple prompts")
    parser.add_argument("--prompt-file", help="UTF-8 file with one input text per line")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kgw-gamma", type=float, default=0.25, help="KGW green-list fraction used by the selected model")
    parser.add_argument("--kgw-bias", type=float, default=2.0, help="KGW logit bias used by the selected model")
    parser.add_argument("--kgw-seeding-scheme", default="auto",
                        choices=("auto", "simple_0", "simple_1", "simple_2"),
                        help="KGW scheme; auto maps k0/k1/k2 to simple_0/simple_1/simple_2")
    parser.add_argument("--output-dir", help="Output directory (default: analysis/result/<family>_<a>_vs_<b>)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--offload-folder", default="analysis/offload",
                        help="Directory for temporary CPU/disk-offloaded model weights")
    parser.add_argument("--hf-token", help="Hugging Face token; otherwise use HF_TOKEN or prompt for Llama")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_model(family, variant):
    repo = MODEL_REPOS[family][variant]
    local = Path("pretrained") / repo.rsplit("/", 1)[-1]
    return str(local) if (local / "config.json").is_file() else repo


def load_tokenizer(source, token, trust_remote_code):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, token=token, use_fast=True,
                                              trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(source, token, device, trust_remote_code, offload_folder):
    from transformers import AutoModelForCausalLM
    kwargs = {"token": token, "low_cpu_mem_usage": True,
              "trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        # device_map=auto may spill layers to disk when GPU/CPU RAM is limited.
        model_offload_dir = Path(offload_folder) / Path(source).name
        model_offload_dir.mkdir(parents=True, exist_ok=True)
        kwargs.update(torch_dtype=torch.float16, device_map="auto",
                      offload_folder=str(model_offload_dir))
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if device.type == "cpu":
        model.to(device)
    model.eval()
    return model


def encode(tokenizer, prompts, max_length, batch_size):
    return [tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            for i in range(0, len(prompts), batch_size)]


def check_tokenization(batches_a, batches_b):
    if len(batches_a) != len(batches_b):
        raise ValueError("The selected models produced different tokenizer batch counts")
    for batch_idx, (a, b) in enumerate(zip(batches_a, batches_b)):
        for key in ("input_ids", "attention_mask"):
            if not torch.equal(a[key], b[key]):
                raise ValueError(f"Tokenizers differ in batch {batch_idx} ({key}); same-context comparison is invalid")


def collect_logits(model, batches):
    input_device = model.get_input_embeddings().weight.device
    outputs = []
    with torch.inference_mode():
        for batch in batches:
            inputs = {key: value.to(input_device) for key, value in batch.items()}
            result = model(**inputs, use_cache=False, return_dict=True)
            last = inputs["attention_mask"].sum(dim=1).long() - 1
            row_ids = torch.arange(last.shape[0], device=input_device)
            outputs.append(result.logits[row_ids, last].float().cpu())
            del result, inputs
    return torch.cat(outputs).numpy()


def release_model(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.prompts:
        prompts = args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as file:
            prompts = [line.rstrip("\n") for line in file if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("No input prompts found")

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.family == "llama" and not token:
        token = getpass.getpass("Llama-2 Hugging Face access token (input hidden): ")
        if not token:
            raise ValueError("A Hugging Face token with Llama-2 access is required")

    source_a = resolve_model(args.family, args.model_a)
    source_b = resolve_model(args.family, args.model_b)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print(f"Family: {args.family}; comparison: {args.model_a} vs {args.model_b}; device: {device}")
    print(f"Model A: {source_a}\nModel B: {source_b}")
    tokenizer_a = load_tokenizer(source_a, token, args.trust_remote_code)
    tokenizer_b = load_tokenizer(source_b, token, args.trust_remote_code)
    batches_a = encode(tokenizer_a, prompts, args.max_length, args.batch_size)
    batches_b = encode(tokenizer_b, prompts, args.max_length, args.batch_size)
    check_tokenization(batches_a, batches_b)

    print(f"Collecting {args.model_a} logits...")
    model = load_model(source_a, token, device, args.trust_remote_code, args.offload_folder)
    logits_a = collect_logits(model, batches_a)
    del model
    release_model(device)

    print(f"Collecting {args.model_b} logits...")
    model = load_model(source_b, token, device, args.trust_remote_code, args.offload_folder)
    logits_b = collect_logits(model, batches_a)
    del model
    release_model(device)

    kgw_scheme = args.kgw_seeding_scheme
    if kgw_scheme == "auto":
        watermark_variant = args.model_b if args.model_b != "base" else args.model_a
        kgw_scheme = {"k0": "simple_0", "k1": "simple_1", "k2": "simple_2"}.get(watermark_variant, "simple_1")

    output_dir = Path(args.output_dir or
                      f"analysis/result/{args.family}_{args.model_a}_vs_{args.model_b}")
    output_dir.mkdir(parents=True, exist_ok=True)
    delta = logits_b - logits_a
    np.save(output_dir / "logits_a.npy", logits_a)
    np.save(output_dir / "logits_b.npy", logits_b)
    np.save(output_dir / "delta_logits.npy", delta)
    # Exact final context IDs are needed to reproduce KGW's context-seeded mask.
    # Batches can have different padded widths, so repad all rows to one width.
    max_width = max(batch["input_ids"].shape[1] for batch in batches_a)
    token_ids = np.full((len(prompts), max_width), tokenizer_a.pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(prompts), max_width), dtype=np.int64)
    row_offset = 0
    for batch in batches_a:
        batch_ids = batch["input_ids"].numpy()
        batch_mask = batch["attention_mask"].numpy()
        width = batch_ids.shape[1]
        batch_rows = batch_ids.shape[0]
        token_ids[row_offset:row_offset + batch_rows, :width] = batch_ids
        attention_mask[row_offset:row_offset + batch_rows, :width] = batch_mask
        row_offset += batch_rows
    np.save(output_dir / "input_ids.npy", token_ids)
    np.save(output_dir / "attention_mask.npy", attention_mask)
    metadata = {"family": args.family, "model_a_variant": args.model_a,
                "model_b_variant": args.model_b, "model_a": source_a, "model_b": source_b,
                "prompts": prompts, "max_length": args.max_length,
                "kgw_gamma": args.kgw_gamma, "kgw_bias": args.kgw_bias,
                "kgw_seeding_scheme": kgw_scheme,
                "definition": "delta_logits = logits_b - logits_a",
                "position": "last non-padding token; next-token logits",
                "mask_context": "full tokenized prompt; KGW greenlist seeded by final context token(s)",
                "shape": list(delta.shape)}
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved arrays to {output_dir.resolve()} (delta shape: {delta.shape})")


if __name__ == "__main__":
    main()
