#!/usr/bin/env python
"""Decompose watermark-vs-clean logit differences into representation/head terms.

For clean model c and watermark model w, with h being the actual vector fed to
lm_head::

    A = Ww @ (hw - hc)
    B = (Ww - Wc) @ hc
    delta_z = A + B

The script compares A, B, and delta_z with the context-seeded KGW direction
(green-list mask minus gamma) for every probe prompt. It loads models itself and
does not modify or depend on ``analyze_logit_diff.py``.

Example::

    python analysis/analyze_logit_decomposition.py \
        --family llama --model-a base --model-b k0 --delta 2
"""

import argparse
import csv
import gc
import getpass
import json
import os
from pathlib import Path

import numpy as np
import torch

BASE_REPOS = {
    "llama": "meta-llama/Llama-2-7b-hf",
    "pythia": "EleutherAI/pythia-1.4b",
}
WATERMARK_REPOS = {
    "llama": "cygu/llama-2-7b-logit-watermark-distill-kgw-{variant}-gamma0.25-delta{delta}",
    "pythia": "cygu/pythia-1.4b-sampling-watermark-distill-kgw-{variant}-gamma0.25-delta{delta}",
}
DEFAULT_PROMPTS = [
    "Explain why the seasons change on Earth in a short paragraph.",
    "A careful scientist records uncertainty instead of hiding it.",
    "Write three practical suggestions for reducing household energy use.",
    "The old railway station stood at the edge of the town, where",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=BASE_REPOS, default="llama")
    parser.add_argument("--model-a", default="base", help="Clean alias, local directory, or Hub repo ID")
    parser.add_argument("--model-b", default="k0", help="Watermark alias, local directory, or Hub repo ID")
    parser.add_argument("--delta", type=int, choices=(1, 2), default=2,
                        help="Delta used for aliases and KGW bias; default: 2")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--seeding-scheme", default="auto",
                        choices=("auto", "simple_0", "simple_1", "simple_2"))
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--offload-folder", default="analysis/offload")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_model(value, family, delta):
    if value == "base":
        repo = BASE_REPOS[family]
    elif value in ("k0", "k1", "k2"):
        repo = WATERMARK_REPOS[family].format(variant=value, delta=delta)
    else:
        return value
    local = Path("pretrained") / repo.rsplit("/", 1)[-1]
    return str(local) if (local / "config.json").is_file() else repo


def read_prompts(args):
    if args.prompts:
        prompts = args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            prompts = [line.rstrip("\n") for line in handle if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("No prompts supplied")
    return prompts


def load_tokenizer(source, token, trust_remote_code):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, token=token, use_fast=True,
                                              trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def encode(tokenizer, prompts, max_length, batch_size):
    return [tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            for i in range(0, len(prompts), batch_size)]


def load_model(source, token, device, trust_remote_code, offload_folder):
    from transformers import AutoModelForCausalLM
    kwargs = {"token": token, "low_cpu_mem_usage": True,
              "trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        offload = Path(offload_folder) / Path(source).name
        offload.mkdir(parents=True, exist_ok=True)
        kwargs.update(torch_dtype=torch.float16, device_map="auto",
                      offload_folder=str(offload))
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.eval()
    if not hasattr(model, "lm_head"):
        raise TypeError(f"{source} does not expose lm_head")
    return model


def collect_logits_and_head_inputs(model, batches):
    """Return next-token logits and actual lm_head inputs for every prompt."""
    input_device = model.get_input_embeddings().weight.device
    captured = []

    def capture_pre_head(_module, module_inputs):
        captured.append(module_inputs[0].detach())

    hook = model.lm_head.register_forward_pre_hook(capture_pre_head)
    logits, head_inputs = [], []
    try:
        with torch.inference_mode():
            for batch in batches:
                inputs = {key: value.to(input_device) for key, value in batch.items()}
                result = model(**inputs, use_cache=False, return_dict=True)
                last = inputs["attention_mask"].sum(dim=1).long() - 1
                rows = torch.arange(last.shape[0], device=last.device)
                logits.append(result.logits[rows, last].float().cpu())
                if not captured:
                    raise RuntimeError("Could not capture lm_head input")
                sequence = captured.pop()
                head_inputs.append(sequence[rows.to(sequence.device), last.to(sequence.device)].float().cpu())
                del result, inputs
    finally:
        hook.remove()
    return torch.cat(logits).numpy(), torch.cat(head_inputs).numpy()


def apply_head(model, vectors):
    """Compute this model's lm_head on vectors from the clean model."""
    weight_device = model.lm_head.weight.device
    dtype = model.lm_head.weight.dtype
    values = torch.as_tensor(vectors, device=weight_device, dtype=dtype)
    with torch.inference_mode():
        return model.lm_head(values).float().cpu().numpy()


def build_green_masks(input_batches, vocab_size, gamma, scheme, tokenizer):
    from watermarks.kgw.watermark_processor import WatermarkBase
    tokenizer_vocab_size = len(tokenizer.get_vocab())
    if tokenizer_vocab_size > vocab_size:
        raise ValueError("Tokenizer vocabulary is larger than model logits vocabulary")
    watermark = WatermarkBase(vocab=list(range(tokenizer_vocab_size)), gamma=gamma,
                              seeding_scheme=scheme, device="cpu")
    if watermark.self_salt:
        raise ValueError("Self-salted KGW schemes are unsupported by this analysis")
    special_ids = set()
    if scheme == "simple_1":
        special_ids = {int(value) for value in
                       (tokenizer.eos_token_id, tokenizer.bos_token_id,
                        tokenizer.pad_token_id, tokenizer.unk_token_id)
                       if value is not None and 0 <= int(value) < vocab_size}
    rows = []
    for batch in input_batches:
        for row in range(batch["input_ids"].shape[0]):
            valid = batch["input_ids"][row][batch["attention_mask"][row].bool()]
            if valid.numel() < watermark.context_width:
                raise ValueError("A prompt is shorter than the KGW context width")
            mask = np.zeros(vocab_size, dtype=bool)
            if not (scheme == "simple_1" and int(valid[-1]) in special_ids):
                green_ids = watermark._get_greenlist_ids(valid.cpu().long()).numpy()
                mask[green_ids] = True
                if special_ids:
                    mask[list(special_ids)] = False
            rows.append(mask)
    return np.asarray(rows, dtype=bool)


def centered_cosine(values, mask_centered):
    values = values.astype(np.float64, copy=False)
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered, axis=1) * np.linalg.norm(mask_centered, axis=1)
    numerator = np.sum(centered * mask_centered, axis=1)
    return np.divide(numerator, denominator, out=np.full(len(values), np.nan), where=denominator > 0)


def save_results(out_dir, rows, arrays, metadata):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "logit_decomposition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "logit_decomposition.npz", **arrays)
    np.save(out_dir / "component_a.npy", arrays["component_a"])
    np.save(out_dir / "component_b.npy", arrays["component_b"])
    np.save(out_dir / "delta_logits.npy", arrays["delta_logits"])
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    prompts = read_prompts(args)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    source_a = resolve_model(args.model_a, args.family, args.delta)
    source_b = resolve_model(args.model_b, args.family, args.delta)
    if any("meta-llama/" in source for source in (source_a, source_b)) and not token:
        token = getpass.getpass("Hugging Face token for gated Llama model (input hidden): ")
        if not token:
            raise ValueError("A Hugging Face token is required")
    scheme = args.seeding_scheme
    if scheme == "auto":
        variant = args.model_b if args.model_b != "base" else args.model_a
        scheme = {"k0": "simple_0", "k1": "simple_1", "k2": "simple_2"}.get(variant, "simple_1")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print(f"Family={args.family}; delta={args.delta}; scheme={scheme}; device={device}")
    print(f"Clean model: {source_a}\nWatermark model: {source_b}")
    tokenizer_a = load_tokenizer(source_a, token, args.trust_remote_code)
    tokenizer_b = load_tokenizer(source_b, token, args.trust_remote_code)
    batches_a = encode(tokenizer_a, prompts, args.max_length, args.batch_size)
    batches_b = encode(tokenizer_b, prompts, args.max_length, args.batch_size)
    if len(batches_a) != len(batches_b):
        raise ValueError("Tokenizers produced different batch counts")
    for index, (batch_a, batch_b) in enumerate(zip(batches_a, batches_b)):
        for key in ("input_ids", "attention_mask"):
            if not torch.equal(batch_a[key], batch_b[key]):
                raise ValueError(f"Tokenizers differ in batch {index} ({key})")
    del tokenizer_b, batches_b

    clean = load_model(source_a, token, device, args.trust_remote_code, args.offload_folder)
    logits_c, h_c = collect_logits_and_head_inputs(clean, batches_a)
    del clean
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    watermarked = load_model(source_b, token, device, args.trust_remote_code, args.offload_folder)
    logits_w, h_w = collect_logits_and_head_inputs(watermarked, batches_a)
    w_on_clean = apply_head(watermarked, h_c)
    del watermarked
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    component_a = logits_w - w_on_clean
    component_b = w_on_clean - logits_c
    delta_logits = logits_w - logits_c
    max_error = float(np.max(np.abs(component_a + component_b - delta_logits)))
    if not np.allclose(component_a + component_b, delta_logits, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"A+B reconstruction failed; max error={max_error}")

    masks = build_green_masks(batches_a, delta_logits.shape[1], args.gamma, scheme, tokenizer_a)
    mask_centered = masks.astype(np.float64) - args.gamma
    cos_a = centered_cosine(component_a, mask_centered)
    cos_b = centered_cosine(component_b, mask_centered)
    cos_delta = centered_cosine(delta_logits, mask_centered)
    rows = []
    for index, prompt in enumerate(prompts):
        rows.append({
            "prompt_index": index, "prompt": prompt,
            "cosine_A_representation_change": float(cos_a[index]),
            "cosine_B_head_change": float(cos_b[index]),
            "cosine_total_delta": float(cos_delta[index]),
            "A_centered_rms": float(np.sqrt(np.mean((component_a[index] - component_a[index].mean()) ** 2))),
            "B_centered_rms": float(np.sqrt(np.mean((component_b[index] - component_b[index].mean()) ** 2))),
            "A_raw_mean": float(component_a[index].mean()),
            "B_raw_mean": float(component_b[index].mean()),
            "reconstruction_max_abs_error": max_error,
        })
    safe_a = Path(args.model_a).name.replace("/", "_")
    safe_b = Path(args.model_b).name.replace("/", "_")
    out_dir = Path(args.output_dir or f"analysis/logit_decomposition_{safe_a}_vs_{safe_b}_delta{args.delta}")
    arrays = {"component_a": component_a, "component_b": component_b,
              "delta_logits": delta_logits, "mask": masks.astype(np.uint8),
              "cosine_A": cos_a, "cosine_B": cos_b, "cosine_delta": cos_delta}
    metadata = {
        "model_a": source_a, "model_b": source_b, "delta": args.delta,
        "gamma": args.gamma, "seeding_scheme": scheme, "prompts": prompts,
        "A": "Ww @ (hw - hc)", "B": "(Ww - Wc) @ hc",
        "h_definition": "actual vector captured immediately before each model's lm_head",
        "max_abs_reconstruction_error": max_error,
    }
    save_results(out_dir, rows, arrays, metadata)
    print("prompt | cos(A, mask) | cos(B, mask) | cos(delta, mask) | max A+B error")
    for row in rows:
        print(f"p{row['prompt_index']} | {row['cosine_A_representation_change']:.6f} | "
              f"{row['cosine_B_head_change']:.6f} | {row['cosine_total_delta']:.6f} | "
              f"{row['reconstruction_max_abs_error']:.3g}")
    print(f"Saved decomposition outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
