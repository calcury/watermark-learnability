#!/usr/bin/env python
"""Analyze KGW alignment of saved same-context model logit differences.

Reads ``analysis/result/...`` outputs from ``analyze_logit_diff.py``, recreates
the context-seeded KGW green mask using this repository's KGW implementation,
and reports (1) centered cosine alignment and (2) green-minus-red mean delta
logit versus the configured KGW bias.

Example::

    !python analysis/analyze_logit_alignment.py analysis/result/llama_base_vs_k0
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Permit execution as `python analysis/analyze_logit_alignment.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", default="analysis/result/llama_base_vs_k0",
                        help="Directory containing delta_logits.npy, input_ids.npy, metadata.json")
    parser.add_argument("--gamma", type=float, default=None, help="Override KGW gamma; otherwise read metadata")
    parser.add_argument("--bias", type=float, default=None, help="Override KGW bias delta; otherwise read metadata")
    parser.add_argument("--seeding-scheme", default=None, help="Override KGW seeding scheme; otherwise read metadata")
    parser.add_argument("--hf-token", default=None, help="Optional token if model tokenizer must be fetched")
    return parser.parse_args()


def read_metadata(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_tokenizer(metadata, token):
    from transformers import AutoTokenizer
    source = metadata.get("model_a")
    if not source:
        raise ValueError("metadata.json is missing model_a; tokenizer is required for KGW special-token handling")
    return AutoTokenizer.from_pretrained(source, token=token, use_fast=True)


def build_green_masks(input_ids, attention_mask, vocab_size, gamma, seeding_scheme, tokenizer):
    from watermarks.kgw.watermark_processor import WatermarkBase

    tokenizer_vocab_size = len(tokenizer.get_vocab()) if tokenizer is not None else vocab_size
    if tokenizer_vocab_size > vocab_size:
        raise ValueError(f"Tokenizer has {tokenizer_vocab_size} IDs but logits expose only {vocab_size} columns")
    base = WatermarkBase(vocab=list(range(tokenizer_vocab_size)), gamma=gamma,
                         seeding_scheme=seeding_scheme, device="cpu")
    if base.self_salt:
        raise ValueError("Self-salted KGW schemes require candidate-token rejection sampling and are not supported by this mask analysis yet")
    special_ids = set()
    if tokenizer is not None and seeding_scheme == "simple_1":
        special_ids = {int(token_id) for token_id in
                       (tokenizer.eos_token_id, tokenizer.bos_token_id,
                        tokenizer.pad_token_id, tokenizer.unk_token_id)
                       if token_id is not None and 0 <= int(token_id) < vocab_size}

    masks = np.zeros((input_ids.shape[0], vocab_size), dtype=bool)
    for row in range(input_ids.shape[0]):
        valid = input_ids[row][attention_mask[row].astype(bool)]
        if valid.size < base.context_width:
            raise ValueError(f"Prompt {row} has only {valid.size} tokens; seeding scheme {seeding_scheme} requires {base.context_width}")
        # KGWWatermark zeros special-token columns, and for simple_1 also
        # zeros the green-list row when the context token itself is special.
        if seeding_scheme == "simple_1" and valid[-1] in special_ids:
            continue
        green_ids = base._get_greenlist_ids(torch.as_tensor(valid, dtype=torch.long))
        masks[row, green_ids.numpy()] = True
        if special_ids:
            masks[row, list(special_ids)] = False
    return masks


def analyze(delta, masks, prompts, gamma, bias):
    results = []
    for i, (values, green) in enumerate(zip(delta, masks)):
        red = ~green
        centered_delta = values - values.mean()
        # Center exactly as the KGW hypothesis specifies: green=1-gamma, red=-gamma.
        centered_mask = green.astype(np.float64) - gamma
        denom = np.linalg.norm(centered_delta) * np.linalg.norm(centered_mask)
        cosine = float(np.dot(centered_delta, centered_mask) / denom) if denom else float("nan")
        green_mean = float(values[green].mean()) if green.any() else float("nan")
        red_mean = float(values[red].mean()) if red.any() else float("nan")
        gap = green_mean - red_mean
        results.append({
            "prompt_index": i, "prompt": prompts[i],
            "delta_mean_raw": float(values.mean()),
            "delta_std_raw": float(values.std()),
            "delta_min_raw": float(values.min()),
            "delta_max_raw": float(values.max()),
            "green_count": int(green.sum()), "red_count": int(red.sum()),
            "green_fraction_actual": float(green.mean()), "gamma_configured": gamma,
            "green_mean_delta": green_mean, "red_mean_delta": red_mean,
            "green_minus_red_delta": gap, "kgw_bias": bias,
            "gap_minus_bias": gap - bias,
            "centered_mask_delta_cosine": cosine,
            "interpretation_cosine_positive": bool(cosine > 0),
        })
    return results


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out_dir, delta, masks, results):
    # Plot observed delta distributions by the actual KGW green/red classification.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, (values, green) in enumerate(zip(delta, masks)):
        edges = np.histogram_bin_edges(values, bins=80)
        axes[0].hist(values[~green], bins=edges, density=True, alpha=.30,
                     label=f"p{i} red")
        axes[0].hist(values[green], bins=edges, density=True, alpha=.85,
                     histtype="step", linewidth=1.5, label=f"p{i} green")
    axes[0].set(title="Delta logits by KGW list", xlabel="delta logit (model B - model A)", ylabel="density")
    axes[0].legend(fontsize=8, ncol=2)

    x = np.arange(len(results))
    gaps = [row["green_minus_red_delta"] for row in results]
    cosines = [row["centered_mask_delta_cosine"] for row in results]
    axes[1].bar(x - .18, gaps, width=.36, label="green mean - red mean")
    axes[1].bar(x + .18, [row["kgw_bias"] for row in results], width=.36, label="configured KGW bias")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(title="Green-red shift vs configured bias", xlabel="prompt index", ylabel="logit shift", xticks=x)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kgw_alignment.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, cosines, color=["#2e8b57" if c > 0 else "#b44" for c in cosines])
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Centered delta-logit / KGW-mask cosine", xlabel="prompt index", ylabel="cosine similarity", xticks=x)
    fig.tight_layout()
    fig.savefig(out_dir / "kgw_mask_cosine.png", dpi=180)
    plt.close(fig)

    # Do not pool prompts in the primary plot: their raw means may differ.
    # Each prompt gets independent bins so its within-prompt structure remains visible.
    n_prompts = len(delta)
    ncols = min(2, n_prompts)
    nrows = (n_prompts + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    for i, (values, green, result) in enumerate(zip(delta, masks, results)):
        ax = axes.flat[i]
        edges = np.histogram_bin_edges(values, bins=120)
        ax.hist(values, bins=edges, density=True, color="#aab2bb", alpha=.35,
                label="All vocabulary tokens")
        ax.hist(values[green], bins=edges, density=True, histtype="step", linewidth=1.8,
                color="#238b45", label="KGW green")
        ax.hist(values[~green], bins=edges, density=True, histtype="step", linewidth=1.8,
                color="#cb3c33", label="KGW red")
        ax.axvline(float(values.mean()), color="#424b54", linestyle="--", linewidth=1,
                   label=f"raw mean={values.mean():.3g}")
        ax.axvline(float(values[green].mean()), color="#238b45", linestyle=":", linewidth=1,
                   label=f"green mean={values[green].mean():.3g}")
        ax.axvline(float(values[~green].mean()), color="#cb3c33", linestyle=":", linewidth=1,
                   label=f"red mean={values[~green].mean():.3g}")
        ax.set(title=(f"Prompt {i}: green-red={result['green_minus_red_delta']:.3g}, "
                     f"cos={result['centered_mask_delta_cosine']:.3g}"),
               xlabel="delta logit (model B - model A), uncentered", ylabel="density")
        ax.legend(fontsize=8)
    for i in range(n_prompts, nrows * ncols):
        axes.flat[i].set_visible(False)
    fig.suptitle("Logit-difference distribution per prompt (raw, uncentered)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_logits_histogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Weighted view: each class density is scaled by its KGW prior fraction,
    # so green + red areas sum to gamma + (1 - gamma) = 1 per prompt.
    gamma = float(results[0]["gamma_configured"]) if results else 0.25
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    for i, (values, green, result) in enumerate(zip(delta, masks, results)):
        ax = axes.flat[i]
        edges = np.histogram_bin_edges(values, bins=120)
        green_density, _ = np.histogram(values[green], bins=edges, density=True)
        red_density, _ = np.histogram(values[~green], bins=edges, density=True)
        ax.hist(values, bins=edges, density=True, color="#aab2bb", alpha=.25,
                label="All vocabulary tokens (area=1)")
        ax.stairs(gamma * green_density, edges, color="#238b45", linewidth=1.8,
                  label=f"KGW green × {gamma:.2f}")
        ax.stairs((1 - gamma) * red_density, edges, color="#cb3c33", linewidth=1.8,
                  label=f"KGW red × {1 - gamma:.2f}")
        ax.set(title=(f"Prompt {i}: weighted green+red area=1, "
                     f"configured gamma={gamma:.2f}"),
               xlabel="delta logit (model B - model A), uncentered", ylabel="weighted density")
        ax.legend(fontsize=8)
    for i in range(n_prompts, nrows * ncols):
        axes.flat[i].set_visible(False)
    fig.suptitle("KGW-weighted logit-difference distributions per prompt", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_logits_histogram_weighted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.input_dir)
    required = ("delta_logits.npy", "input_ids.npy", "attention_mask.npy", "metadata.json")
    missing = [name for name in required if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} under {out_dir}. Re-run analyze_logit_diff.py to export exact token IDs.")

    delta = np.load(out_dir / "delta_logits.npy", allow_pickle=False).astype(np.float64)
    input_ids = np.load(out_dir / "input_ids.npy", allow_pickle=False)
    attention_mask = np.load(out_dir / "attention_mask.npy", allow_pickle=False)
    metadata = read_metadata(out_dir / "metadata.json")
    if delta.ndim != 2 or input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Expected 2-D delta/input arrays and matching input_ids/attention_mask shapes")
    if delta.shape[0] != input_ids.shape[0] or delta.shape[1] != int(metadata.get("shape", delta.shape)[-1]):
        raise ValueError("Saved logits and input context arrays do not align")

    gamma = args.gamma if args.gamma is not None else float(metadata.get("kgw_gamma", .25))
    bias = args.bias if args.bias is not None else float(metadata.get("kgw_bias", 2.0))
    seeding_scheme = args.seeding_scheme or metadata.get("kgw_seeding_scheme", "simple_1")
    tokenizer = load_tokenizer(metadata, args.hf_token)
    masks = build_green_masks(input_ids, attention_mask, delta.shape[1],
                              gamma, seeding_scheme, tokenizer)
    prompts = metadata.get("prompts", [f"prompt_{i}" for i in range(delta.shape[0])])
    rows = analyze(delta, masks, prompts, gamma, bias)
    np.save(out_dir / "kgw_green_masks.npy", masks.astype(np.uint8))
    np.savez_compressed(
        out_dir / "kgw_metrics.npz",
        centered_mask_delta_cosine=np.asarray([r["centered_mask_delta_cosine"] for r in rows]),
        green_mean_delta=np.asarray([r["green_mean_delta"] for r in rows]),
        red_mean_delta=np.asarray([r["red_mean_delta"] for r in rows]),
        green_minus_red_delta=np.asarray([r["green_minus_red_delta"] for r in rows]),
        kgw_bias=np.asarray([r["kgw_bias"] for r in rows]),
        gap_minus_bias=np.asarray([r["gap_minus_bias"] for r in rows]),
        green_counts=np.asarray([r["green_count"] for r in rows]),
        red_counts=np.asarray([r["red_count"] for r in rows]),
    )
    save_csv(out_dir / "kgw_alignment.csv", rows)
    make_plots(out_dir, delta, masks, rows)

    print(f"KGW settings: gamma={gamma}, bias={bias}, seeding_scheme={seeding_scheme}")
    print(f"Saved green masks: {out_dir / 'kgw_green_masks.npy'}")
    print(f"Saved alignment table: {out_dir / 'kgw_alignment.csv'}")
    print(f"Saved plots: {out_dir / 'kgw_alignment.png'}, {out_dir / 'kgw_mask_cosine.png'}, "
          f"{out_dir / 'delta_logits_histogram.png'} (original per-prompt), "
          f"{out_dir / 'delta_logits_histogram_weighted.png'} (gamma-weighted per-prompt)")
    print(f"{'p':>4} {'cosine':>10} {'green mean':>12} {'red mean':>12} {'gap':>10} {'bias':>10} {'gap-bias':>10}")
    for row in rows:
        print(f"p{row['prompt_index']:>3} {row['centered_mask_delta_cosine']:10.5f} "
              f"{row['green_mean_delta']:12.5f} {row['red_mean_delta']:12.5f} "
              f"{row['green_minus_red_delta']:10.5f} {row['kgw_bias']:10.5f} {row['gap_minus_bias']:10.5f}")
    print("Interpretation: cosine > 0 indicates alignment; gap near bias is consistent with the configured offset, not by itself causal proof.")


if __name__ == "__main__":
    main()
