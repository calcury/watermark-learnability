#!/usr/bin/env python
"""Compare every hidden-state layer in model A against every layer in model B.

For each layer pair, reports mean paired-token cosine similarity (when hidden
sizes match) and linear CKA. Writes the full layer-pair table and heatmaps.

Examples::

    python analysis/compare_cross_layer.py --family llama --model-a base --model-b k0 --delta 2
    python analysis/compare_cross_layer.py --model-a /models/model_a --model-b org/model-b
"""

import argparse
import csv
import getpass
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

BASE_REPOS = {
    "llama": "meta-llama/Llama-2-7b-hf",
    "pythia": "EleutherAI/pythia-1.4b",
}
WATERMARK_REPO_TEMPLATES = {
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
    parser.add_argument("--family", choices=BASE_REPOS, default="llama",
                        help="Model family for resolving base/k0/k1/k2 aliases")
    parser.add_argument("--model-a", default="base",
                        help="Model alias (base/k0/k1/k2), local model directory, or Hub repo ID")
    parser.add_argument("--model-b", default="k0",
                        help="Model alias (base/k0/k1/k2), local model directory, or Hub repo ID")
    parser.add_argument("--delta", type=int, choices=(1, 2), default=2,
                        help="KGW delta variant for aliases; default is delta2")
    parser.add_argument("--prompt", dest="prompts", action="append",
                        help="Probe prompt; may be repeated (default: four built-in prompts)")
    parser.add_argument("--prompt-file", help="UTF-8 file containing one prompt per line")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum aligned valid token positions used for metrics (default: 512)")
    parser.add_argument("--output-dir", default=None,
                        help="Output folder (default: analysis/cross_layer_<A>_vs_<B>_delta<N>)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token; otherwise use HF_TOKEN or securely prompt for gated Llama")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_model(value, family, delta):
    """Resolve a known alias; leave arbitrary local paths/Hub IDs untouched."""
    if value in ("base", "k0", "k1", "k2"):
        if value == "base":
            repo = BASE_REPOS[family]
        else:
            if delta == 1 and value == "k2":
                raise ValueError("No default k2-delta1 checkpoint is configured; pass its explicit Hub ID or local path.")
            repo = WATERMARK_REPO_TEMPLATES[family].format(variant=value, delta=delta)
        local = Path("pretrained") / repo.rsplit("/", 1)[-1]
        return str(local) if (local / "config.json").is_file() else repo
    return value


def read_prompts(args):
    if args.prompts:
        prompts = args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if not prompts:
        raise ValueError("No non-empty probe prompts supplied")
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
    batches = []
    for start in range(0, len(prompts), batch_size):
        batches.append(tokenizer(prompts[start:start + batch_size], return_tensors="pt",
                                 padding=True, truncation=True, max_length=max_length))
    return batches


def load_hidden_states(source, batches, token, device, trust_remote_code):
    from transformers import AutoModelForCausalLM
    kwargs = {"token": token, "low_cpu_mem_usage": True,
              "trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        offload = Path("analysis/offload") / Path(source).name
        offload.mkdir(parents=True, exist_ok=True)
        kwargs.update(torch_dtype=torch.float16, device_map="auto", offload_folder=str(offload))
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.eval()
    layer_chunks = None
    with torch.inference_mode():
        for batch in batches:
            embedding_device = model.get_input_embeddings().weight.device
            inputs = {key: value.to(embedding_device) for key, value in batch.items()}
            output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            valid = inputs["attention_mask"].bool()
            if layer_chunks is None:
                layer_chunks = [[] for _ in output.hidden_states]
            for index, hidden in enumerate(output.hidden_states):
                layer_chunks[index].append(hidden[valid].float().cpu())
            del output, inputs
    states = [torch.cat(chunks, dim=0).numpy() for chunks in layer_chunks]
    config = model.config
    model_info = {
        "source": source,
        "architecture": config.model_type,
        "hidden_size": int(config.hidden_size),
        "transformer_layers": int(config.num_hidden_layers),
        "hidden_state_count": len(states),
    }
    del model
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return states, model_info


def paired_cosine_similarity(x, y):
    if x.shape[1] != y.shape[1]:
        return None
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    xn = np.linalg.norm(x, axis=1).clip(min=1e-12)
    yn = np.linalg.norm(y, axis=1).clip(min=1e-12)
    return float(np.mean(np.sum(x * y, axis=1) / (xn * yn)))


def centered_sample_gram(features):
    """Return centered linear Gram matrix, equivalent for linear CKA."""
    features = features.astype(np.float64, copy=False)
    features = features - features.mean(axis=0, keepdims=True)
    return features @ features.T


def linear_cka_from_grams(gram_x, gram_y):
    denominator = np.linalg.norm(gram_x, "fro") * np.linalg.norm(gram_y, "fro")
    return float(np.sum(gram_x * gram_y) / denominator) if denominator > 0 else float("nan")


def compare_all_layer_pairs(states_a, states_b, max_tokens):
    sample_count = min(min(state.shape[0] for state in states_a),
                       min(state.shape[0] for state in states_b))
    if sample_count == 0:
        raise ValueError("No valid (non-padding) token representations were collected")
    if any(state.shape[0] != sample_count for state in states_a + states_b):
        raise ValueError("Layer representations have mismatched sample counts")
    if sample_count > max_tokens:
        indices = np.linspace(0, sample_count - 1, max_tokens, dtype=np.int64)
        states_a = [state[indices] for state in states_a]
        states_b = [state[indices] for state in states_b]
    gram_a = [centered_sample_gram(state) for state in states_a]
    gram_b = [centered_sample_gram(state) for state in states_b]
    rows = []
    cosine_matrix = np.full((len(states_a), len(states_b)), np.nan)
    cka_matrix = np.full_like(cosine_matrix, np.nan)
    for layer_a, x in enumerate(states_a):
        for layer_b, y in enumerate(states_b):
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Unaligned token samples at layers A{layer_a}, B{layer_b}")
            cosine = paired_cosine_similarity(x, y)
            cka = linear_cka_from_grams(gram_a[layer_a], gram_b[layer_b])
            cosine_matrix[layer_a, layer_b] = np.nan if cosine is None else cosine
            cka_matrix[layer_a, layer_b] = cka
            rows.append({
                "layer_a": layer_a, "layer_b": layer_b,
                "cosine_similarity": "" if cosine is None else cosine,
                "linear_cka": cka,
                "tokens": int(x.shape[0]),
                "hidden_size_a": int(x.shape[1]), "hidden_size_b": int(y.shape[1]),
            })
    return rows, cosine_matrix, cka_matrix, int(states_a[0].shape[0])


def save_outputs(rows, cosine, cka, out_dir, metadata):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_layer_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "cross_layer_matrices.npz",
                        cosine_similarity=cosine, linear_cka=cka)
    (out_dir / "experiment_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(max(12, cosine.shape[1] * .55),
                                           max(5, cosine.shape[0] * .42)))
    for ax, matrix, title, cmap, vmin, vmax in (
        (axes[0], cosine, "Paired-token cosine similarity", "coolwarm", -1, 1),
        (axes[1], cka, "Linear CKA", "viridis", 0, 1),
    ):
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Model B hidden-state layer")
        ax.set_ylabel("Model A hidden-state layer")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    plot_path = out_dir / "cross_layer_heatmaps.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return csv_path, out_dir / "cross_layer_matrices.npz", plot_path


def main():
    args = parse_args()
    prompts = read_prompts(args)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    source_a = resolve_model(args.model_a, args.family, args.delta)
    source_b = resolve_model(args.model_b, args.family, args.delta)
    if any("meta-llama/" in source for source in (source_a, source_b)) and not token:
        token = getpass.getpass("Hugging Face token for gated Llama model (input hidden): ")
        if not token:
            raise ValueError("A Hugging Face token with Llama-2 access is required")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print(f"Family={args.family}; delta={args.delta}; device={device}")
    print(f"Model A: {source_a}\nModel B: {source_b}\nProbe prompts: {len(prompts)}")
    tokenizer_a = load_tokenizer(source_a, token, args.trust_remote_code)
    tokenizer_b = load_tokenizer(source_b, token, args.trust_remote_code)
    batches_a = encode(tokenizer_a, prompts, args.max_length, args.batch_size)
    batches_b = encode(tokenizer_b, prompts, args.max_length, args.batch_size)
    if len(batches_a) != len(batches_b):
        raise ValueError("The two tokenizers generated different batch counts")
    for index, (batch_a, batch_b) in enumerate(zip(batches_a, batches_b)):
        for key in ("input_ids", "attention_mask"):
            if not torch.equal(batch_a[key], batch_b[key]):
                raise ValueError(f"Tokenizers differ in batch {index} ({key}); layer comparison requires identical token positions")
    del tokenizer_b, batches_b

    states_a, info_a = load_hidden_states(source_a, batches_a, token, device, args.trust_remote_code)
    states_b, info_b = load_hidden_states(source_b, batches_a, token, device, args.trust_remote_code)
    if info_a["hidden_size"] != info_b["hidden_size"]:
        print("Note: hidden sizes differ; cosine is left blank for incompatible layer pairs, but CKA remains available.")
    rows, cosine_matrix, cka_matrix, used_tokens = compare_all_layer_pairs(
        states_a, states_b, args.max_tokens)
    safe_a = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_a.rsplit("/", 1)[-1])
    safe_b = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_b.rsplit("/", 1)[-1])
    out_dir = Path(args.output_dir or
                   f"analysis/cross_layer_{safe_a}_vs_{safe_b}_delta{args.delta}")
    metadata = {
        "family": args.family, "delta": args.delta,
        "model_a": info_a, "model_b": info_b,
        "prompts": prompts, "max_length": args.max_length,
        "valid_tokens_used": used_tokens,
        "cosine_definition": "mean paired-token cosine similarity; blank if hidden sizes differ",
        "cka_definition": "centered linear CKA over aligned valid-token rows",
        "layer_index": "0 is embedding output; 1..N are transformer block outputs",
    }
    csv_path, matrix_path, plot_path = save_outputs(rows, cosine_matrix, cka_matrix, out_dir, metadata)
    print(f"Compared {len(states_a)} x {len(states_b)} hidden-state layers using {used_tokens} token positions.")
    print(f"Saved layer-pair metrics: {csv_path}")
    print(f"Saved matrices: {matrix_path}\nSaved heatmaps: {plot_path}")
    print("Layer A | Layer B | cosine similarity | linear CKA")
    for row in rows:
        cosine_text = "N/A" if row["cosine_similarity"] == "" else f"{row['cosine_similarity']:.5f}"
        print(f"{row['layer_a']:7d} | {row['layer_b']:7d} | {cosine_text:17s} | {row['linear_cka']:.5f}")


if __name__ == "__main__":
    main()
