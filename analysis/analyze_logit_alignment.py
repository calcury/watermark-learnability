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
    parser.add_argument("input_dir", nargs="?", default=None,
                        help="Directory containing delta_logits.npy, input_ids.npy, metadata.json")
    parser.add_argument("--gamma", type=float, default=None, help="Override KGW gamma; otherwise read metadata")
    parser.add_argument("--bias", type=float, default=None, help="Override KGW bias; otherwise read metadata (or --delta)")
    parser.add_argument("--delta", type=int, choices=(1, 2), default=None,
                        help="Expected KGW delta (checks metadata); supplies the bias only when metadata lacks it")
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


def save_token_level_csv(path, delta, masks, gamma, logits_a=None, logits_b=None, prompts=None):
    """Write one auditable row per prompt and vocabulary token."""
    fields = ["prompt_index", "prompt", "token_id", "kgw_class", "is_green",
              "logits_a", "logits_b", "delta_logit", "delta_centered",
              "centered_mask_value"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for prompt_index, (values, green) in enumerate(zip(delta, masks)):
            centered = values - values.mean()
            prompt = prompts[prompt_index] if prompts and prompt_index < len(prompts) else f"prompt_{prompt_index}"
            for token_id, (value, is_green) in enumerate(zip(values, green)):
                writer.writerow({
                    "prompt_index": prompt_index, "prompt": prompt, "token_id": token_id,
                    "kgw_class": "green" if is_green else "red", "is_green": int(is_green),
                    "logits_a": "" if logits_a is None else float(logits_a[prompt_index, token_id]),
                    "logits_b": "" if logits_b is None else float(logits_b[prompt_index, token_id]),
                    "delta_logit": float(value), "delta_centered": float(centered[token_id]),
                    "centered_mask_value": (1.0 - gamma) if is_green else -gamma,
                })


def save_summary_json(path, metadata, rows, delta, masks, gamma, bias, seeding_scheme):
    """Save auditable run settings and descriptive aggregates without token-independence claims."""
    green_values = [values[mask] for values, mask in zip(delta, masks)]
    red_values = [values[~mask] for values, mask in zip(delta, masks)]
    pooled_green = np.concatenate(green_values)
    pooled_red = np.concatenate(red_values)
    metric_names = ("centered_mask_delta_cosine", "green_minus_red_delta",
                    "gap_minus_bias", "delta_mean_raw", "delta_std_raw")
    aggregates = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        aggregates[name] = {
            "mean_across_prompts": float(np.mean(values)),
            "median_across_prompts": float(np.median(values)),
            "std_across_prompts_population": float(np.std(values)),
            "min_across_prompts": float(np.min(values)),
            "max_across_prompts": float(np.max(values)),
        }
    summary = {
        "experiment": {
            "family": metadata.get("family"),
            "model_a_variant": metadata.get("model_a_variant"),
            "model_b_variant": metadata.get("model_b_variant"),
            "model_a": metadata.get("model_a"), "model_b": metadata.get("model_b"),
            "delta_definition": metadata.get("definition", "logits_b - logits_a"),
            "logit_position": metadata.get("position"),
            "kgw_gamma": gamma, "kgw_bias": bias, "kgw_seeding_scheme": seeding_scheme,
            "prompt_count": int(delta.shape[0]), "vocab_size": int(delta.shape[1]),
            "prompts": metadata.get("prompts", []),
        },
        "pooled_descriptive_metrics": {
            "green_count": int(sum(mask.sum() for mask in masks)),
            "red_count": int(sum((~mask).sum() for mask in masks)),
            "green_fraction_actual": float(np.mean(masks)),
            "green_mean_delta": float(pooled_green.mean()),
            "red_mean_delta": float(pooled_red.mean()),
            "green_minus_red_delta": float(pooled_green.mean() - pooled_red.mean()),
            "configured_bias": float(bias),
            "gap_minus_bias": float(pooled_green.mean() - pooled_red.mean() - bias),
            "delta_mean_raw_all_prompt_tokens": float(delta.mean()),
            "delta_std_raw_all_prompt_tokens": float(delta.std()),
        },
        "across_prompt_descriptives": aggregates,
        "per_prompt_metrics": rows,
        "interpretation_note": (
            "Across-prompt values are descriptive for this probe set. Vocabulary entries are not treated as independent samples;"
            " no inferential p-values are reported. A positive centered cosine indicates directional alignment."
        ),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


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
    if args.input_dir is None:
        delta_tag = f"_delta{args.delta}" if args.delta is not None else "_delta2"
        default_dir = Path(f"analysis/result/llama_base_vs_k0{delta_tag}")
        legacy_delta2_dir = Path("analysis/result/llama_base_vs_k0")
        if args.delta is None and not default_dir.exists() and legacy_delta2_dir.exists():
            default_dir = legacy_delta2_dir
        args.input_dir = str(default_dir)
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

    logits_a_path, logits_b_path = out_dir / "logits_a.npy", out_dir / "logits_b.npy"
    logits_a = np.load(logits_a_path, allow_pickle=False) if logits_a_path.is_file() else None
    logits_b = np.load(logits_b_path, allow_pickle=False) if logits_b_path.is_file() else None
    if (logits_a is None) != (logits_b is None):
        raise ValueError("Both logits_a.npy and logits_b.npy must be present together")
    if logits_a is not None:
        if logits_a.shape != delta.shape or logits_b.shape != delta.shape:
            raise ValueError("Saved model logits do not match delta_logits shape")
        if not np.allclose(logits_b.astype(np.float64) - logits_a.astype(np.float64), delta, rtol=1e-5, atol=1e-6):
            raise ValueError("delta_logits.npy does not match logits_b.npy - logits_a.npy")

    metadata_delta = metadata.get("kgw_delta")
    if metadata_delta is None:
        import re
        model_ref = str(metadata.get("model_b", ""))
        match = re.search(r"-delta([12])(?:$|[-/])", model_ref)
        metadata_delta = int(match.group(1)) if match else None
    if args.delta is not None and metadata_delta is not None and int(metadata_delta) != args.delta:
        raise ValueError(f"Requested --delta {args.delta}, but result metadata indicates delta{metadata_delta}")
    gamma = args.gamma if args.gamma is not None else float(metadata.get("kgw_gamma", .25))
    default_bias = float(args.delta if args.delta is not None else (metadata_delta or 2))
    if args.bias is not None:
        bias = float(args.bias)
    elif args.delta is not None:
        bias = float(args.delta)
    else:
        bias = float(metadata.get("kgw_bias", default_bias))
    seeding_scheme = args.seeding_scheme or metadata.get("kgw_seeding_scheme", "simple_1")
    mask_path = out_dir / "kgw_green_masks.npy"
    mask_metadata_path = out_dir / "kgw_green_masks_metadata.json"
    expected_mask_metadata = {"kgw_gamma": gamma, "kgw_seeding_scheme": seeding_scheme,
                              "shape": list(delta.shape), "mask_context": metadata.get("mask_context")}
    reuse_mask = False
    if mask_path.is_file():
        saved_mask_metadata = None
        if mask_metadata_path.is_file():
            saved_mask_metadata = read_metadata(mask_metadata_path)
        else:
            # Legacy masks were generated with the run-level settings; only
            # trust them when no analysis-time override changes those settings.
            run_gamma = float(metadata.get("kgw_gamma", .25))
            run_scheme = metadata.get("kgw_seeding_scheme", "simple_1")
            if gamma == run_gamma and seeding_scheme == run_scheme:
                saved_mask_metadata = expected_mask_metadata
        reuse_mask = saved_mask_metadata == expected_mask_metadata
        if reuse_mask:
            masks = np.load(mask_path, allow_pickle=False).astype(bool)
            if masks.shape != delta.shape:
                raise ValueError(f"Saved KGW mask shape {masks.shape} does not match logits {delta.shape}")
            print(f"Reusing KGW mask verified for gamma={gamma}, scheme={seeding_scheme}: {mask_path}")
    if not reuse_mask:
        tokenizer = load_tokenizer(metadata, args.hf_token)
        masks = build_green_masks(input_ids, attention_mask, delta.shape[1],
                                  gamma, seeding_scheme, tokenizer)
    mask_metadata_path.write_text(json.dumps(expected_mask_metadata, indent=2), encoding="utf-8")
    prompts = metadata.get("prompts", [f"prompt_{i}" for i in range(delta.shape[0])])
    if len(prompts) != delta.shape[0]:
        raise ValueError("Prompt count in metadata does not match saved logits")
    rows = analyze(delta, masks, prompts, gamma, bias)
    np.save(mask_path, masks.astype(np.uint8))
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
        green_fraction_actual=np.asarray([r["green_fraction_actual"] for r in rows]),
        delta_mean_raw=np.asarray([r["delta_mean_raw"] for r in rows]),
        delta_std_raw=np.asarray([r["delta_std_raw"] for r in rows]),
        delta_min_raw=np.asarray([r["delta_min_raw"] for r in rows]),
        delta_max_raw=np.asarray([r["delta_max_raw"] for r in rows]),
    )
    save_csv(out_dir / "kgw_alignment.csv", rows)
    save_token_level_csv(out_dir / "kgw_token_level.csv", delta, masks, gamma,
                         logits_a=logits_a, logits_b=logits_b, prompts=prompts)
    save_summary_json(out_dir / "kgw_summary.json", metadata, rows, delta, masks,
                      gamma, bias, seeding_scheme)
    make_plots(out_dir, delta, masks, rows)
    obsolete_pooled_plot = out_dir / "delta_logits_histogram_pooled.png"
    if obsolete_pooled_plot.is_file():
        obsolete_pooled_plot.unlink()

    print(f"KGW settings: gamma={gamma}, bias={bias}, seeding_scheme={seeding_scheme}")
    print(f"Saved prompt metrics: {out_dir / 'kgw_alignment.csv'}")
    print(f"Saved token-level data: {out_dir / 'kgw_token_level.csv'}")
    print(f"Saved experiment summary: {out_dir / 'kgw_summary.json'}")
    print(f"Saved plots: {out_dir / 'kgw_alignment.png'}, {out_dir / 'kgw_mask_cosine.png'}, "
          f"{out_dir / 'delta_logits_histogram.png'}, {out_dir / 'delta_logits_histogram_weighted.png'}")
    print(f"{'p':>4} {'cosine':>10} {'green mean':>12} {'red mean':>12} {'gap':>10} {'bias':>10} {'gap-bias':>10}")
    for row in rows:
        print(f"p{row['prompt_index']:>3} {row['centered_mask_delta_cosine']:10.5f} "
              f"{row['green_mean_delta']:12.5f} {row['red_mean_delta']:12.5f} "
              f"{row['green_minus_red_delta']:10.5f} {row['kgw_bias']:10.5f} {row['gap_minus_bias']:10.5f}")
    print("Interpretation: cosine > 0 indicates alignment; gap near bias is consistent with the configured offset, not by itself causal proof.")


if __name__ == "__main__":
    main()
