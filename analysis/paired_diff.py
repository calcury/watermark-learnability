#!/usr/bin/env python
"""Paired B1/B2 (and optional A1/A2) representation experiment.

All models must be local directories. The script tokenizes each prompt once,
asserts both models receive identical token IDs, loads models sequentially, and
writes per-group/per-layer distribution statistics and plots.

Example from Colab::

    !python analysis/paired_diff.py \
        --b1 pretrained/pythia-1.4b \
        --b2 pretrained/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2 \
        --prompt-file data/probe.tsv

``probe.tsv`` may contain one prompt per line, or ``group<TAB>prompt``. Groups
should be fixed before looking at results (for example clean, trigger, random).
"""

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

DEFAULT_B1 = "pretrained/pythia-1.4b"
DEFAULT_B2 = "pretrained/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2"
DEFAULT_PROMPTS = [
    "The history of science is a story of people asking questions about the world.",
    "A good education helps people understand their communities and make informed decisions.",
    "In the future, renewable energy may transform how cities are designed.",
    "The recipe calls for fresh vegetables, olive oil, and a little salt.",
    "When building reliable software, testing small components can prevent larger problems.",
]


def args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b1", default=DEFAULT_B1, help="Normal student B1 local directory")
    p.add_argument("--b2", default=DEFAULT_B2, help="Watermark student B2 local directory")
    p.add_argument("--a1", default=None, help="Optional normal teacher A1 local directory")
    p.add_argument("--a2", default=None, help="Optional watermarked teacher A2 local directory")
    p.add_argument("--prompt-file", help="UTF-8 lines: prompt, or group<TAB>prompt")
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--max-prompts", type=int, default=1000)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-hidden-tokens", type=int, default=4096, help="Cap activation samples to bound Colab RAM use")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for 95%% CI; 0 disables")
    p.add_argument("--max-cka-tokens", type=int, default=512, help="Tokens sampled per group for CKA")
    p.add_argument("--output-dir", default="analysis/paired_diff_output")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_prompts(ns):
    if ns.prompts:
        return [("default", text) for text in ns.prompts[: ns.max_prompts]]
    if ns.prompt_file:
        rows = []
        with open(ns.prompt_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                rows.append((parts[0], parts[1]) if len(parts) == 2 else ("default", line))
                if len(rows) >= ns.max_prompts:
                    break
        if rows:
            return rows
    return [("clean", text) for text in DEFAULT_PROMPTS[: ns.max_prompts]]


def local_dir(value, name):
    path = Path(value)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"{name} is not a local model directory with config.json: {path}")
    return path


def tokenize(tokenizer, rows, max_length, batch_size):
    batches = []
    for start in range(0, len(rows), batch_size):
        texts = [text for _, text in rows[start : start + batch_size]]
        batches.append(tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length))
    return batches


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, device, trust_remote_code, max_hidden_tokens):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    model._analysis_max_hidden_tokens = max_hidden_tokens
    return model


def collect(model, batches, device):
    layers = [[] for _ in range(model.config.num_hidden_layers + 1)]
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        mask = inputs["attention_mask"].bool()
        for i, state in enumerate(out.hidden_states):
            layers[i].append(state[mask].float().cpu())
        del out, inputs
    arrays = [torch.cat(x).numpy() for x in layers]
    cap = getattr(model, "_analysis_max_hidden_tokens", None)
    if cap and arrays[0].shape[0] > cap:
        indices = np.linspace(0, arrays[0].shape[0] - 1, cap, dtype=np.int64)
        arrays = [arr[indices] for arr in arrays]
    return arrays


def cka(x, y, max_tokens):
    if len(x) > max_tokens:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), max_tokens, replace=False)
        x, y = x[idx], y[idx]
    x = x.astype(np.float64) - x.mean(0, keepdims=True)
    y = y.astype(np.float64) - y.mean(0, keepdims=True)
    xx, yy, xy = x.T @ x, y.T @ y, x.T @ y
    den = np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro")
    return float(np.sum(xy * xy) / den) if den > 0 else float("nan")


def stats(x, y, bootstrap):
    x, y = x.astype(np.float64), y.astype(np.float64)
    xn = np.linalg.norm(x, axis=1).clip(min=1e-12)
    yn = np.linalg.norm(y, axis=1).clip(min=1e-12)
    cosine = 1.0 - np.sum(x * y, axis=1) / (xn * yn)
    relative_l2 = np.linalg.norm(x - y, axis=1) / (yn * np.sqrt(x.shape[1]) + 1e-12)
    result = {
        "cosine_mean": float(cosine.mean()), "cosine_std": float(cosine.std()),
        "cosine_p50": float(np.quantile(cosine, .5)), "cosine_p90": float(np.quantile(cosine, .9)),
        "relative_l2_mean": float(relative_l2.mean()),
        "student_rms": float(np.sqrt(np.mean(x * x))), "reference_rms": float(np.sqrt(np.mean(y * y))),
        "tokens": int(len(x)),
    }
    if bootstrap and len(cosine) > 1:
        rng = np.random.default_rng(12345)
        means = np.empty(bootstrap)
        for i in range(bootstrap):
            means[i] = cosine[rng.integers(0, len(cosine), len(cosine))].mean()
        result["cosine_ci95_low"], result["cosine_ci95_high"] = map(float, np.quantile(means, [.025, .975]))
    else:
        result["cosine_ci95_low"] = result["cosine_ci95_high"] = result["cosine_mean"]
    return result


def compare_pair(name, left, right, groups, bootstrap, max_cka_tokens):
    rows = []
    group_ids = sorted(set(groups))
    for layer, (x, y) in enumerate(zip(left, right)):
        for group in group_ids:
            idx = np.array([g == group for g in groups])
            if not idx.any():
                continue
            row = {"pair": name, "group": group, "layer": layer}
            row.update(stats(x[idx], y[idx], bootstrap))
            row["linear_cka"] = cka(x[idx], y[idx], max_cka_tokens)
            rows.append(row)
    return rows


def save(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_layer_metrics.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": len(rows), "description": "Paired same-token layerwise comparison"}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = sorted(set(r["pair"] for r in rows))
    groups = sorted(set(r["group"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for pair in pairs:
        for group in groups:
            subset = [r for r in rows if r["pair"] == pair and r["group"] == group]
            if subset:
                label = f"{pair}:{group}"
                axes[0].plot([r["layer"] for r in subset], [r["cosine_mean"] for r in subset], marker="o", label=label)
                axes[1].plot([r["layer"] for r in subset], [r["linear_cka"] for r in subset], marker="o", label=label)
    axes[0].set_title("Same-token cosine distance")
    axes[1].set_title("Linear CKA (higher = more similar)")
    for ax in axes:
        ax.set_xlabel("Layer (0 = embedding)"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); plot_path = output_dir / "paired_layer_metrics.png"; fig.savefig(plot_path, dpi=160); plt.close(fig)
    return csv_path, plot_path


def main():
    ns = args()
    rows = read_prompts(ns)
    b1, b2 = local_dir(ns.b1, "--b1"), local_dir(ns.b2, "--b2")
    device = torch.device("cuda" if ns.device == "auto" and torch.cuda.is_available() else ns.device if ns.device != "auto" else "cpu")
    print(f"Device: {device}; prompts: {len(rows)}")
    tok1, tok2 = load_tokenizer(b1), load_tokenizer(b2)
    batches = tokenize(tok1, rows, ns.max_length, ns.batch_size)
    batches2 = tokenize(tok2, rows, ns.max_length, ns.batch_size)
    for a, b in zip(batches, batches2):
        if not torch.equal(a["input_ids"], b["input_ids"]) or not torch.equal(a["attention_mask"], b["attention_mask"]):
            raise RuntimeError("B1 and B2 tokenizers produce different IDs; use a shared tokenizer or aligned token IDs.")
    token_groups = []
    for batch_index, batch in enumerate(batches):
        for row_index, (group, _) in enumerate(rows[batch_index * ns.batch_size : (batch_index + 1) * ns.batch_size]):
            token_groups.extend([group] * int(batch["attention_mask"][row_index].sum()))
    if len(token_groups) > ns.max_hidden_tokens:
        keep = np.linspace(0, len(token_groups) - 1, ns.max_hidden_tokens, dtype=np.int64)
        token_groups = [token_groups[i] for i in keep]
    del tok1, tok2, batches2
    print("Loading B1...")
    m1 = load_model(b1, device, ns.trust_remote_code, ns.max_hidden_tokens); h1 = collect(m1, batches, device)
    del m1; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    print("Loading B2...")
    m2 = load_model(b2, device, ns.trust_remote_code, ns.max_hidden_tokens); h2 = collect(m2, batches, device)
    del m2; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    # token_groups follows the unpadded token order used by collect().
    all_rows = compare_pair("B1_vs_B2", h1, h2, token_groups, ns.bootstrap, ns.max_cka_tokens)
    if ns.a1 and ns.a2:
        a1, a2 = local_dir(ns.a1, "--a1"), local_dir(ns.a2, "--a2")
        print("Loading optional A1/A2 teacher pair...")
        ma1 = load_model(a1, device, ns.trust_remote_code, ns.max_hidden_tokens)
        ha1 = collect(ma1, batches, device)
        del ma1; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
        ma2 = load_model(a2, device, ns.trust_remote_code, ns.max_hidden_tokens)
        ha2 = collect(ma2, batches, device)
        del ma2; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
        # Teacher activations are collected sequentially; compare them after both runs.
        all_rows.extend(compare_pair("A1_vs_A2", ha1, ha2, token_groups, ns.bootstrap, ns.max_cka_tokens))
    csv_path, plot_path = save(all_rows, Path(ns.output_dir))
    print(f"Saved metrics: {csv_path}\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
