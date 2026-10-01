#!/usr/bin/env python
"""Analyze layer-by-layer top-token propagation from logit-lens NPY outputs.

This script asks whether watermark-related divergence is gradual across layers
or concentrated at the final transition. It compares base and watermarked model
trajectories for each retained token position.

Example::

    python analysis/analyze_logit_lens_propagation.py \
        analysis/logit_lens_base_vs_k0_delta2
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir")
    p.add_argument("--top-k", type=int, default=None,
                   help="Use the first K saved ranks; default uses all saved ranks")
    p.add_argument("--output-dir", default=None,
                   help="Output folder; default writes beside the input NPY files")
    return p.parse_args()


def load_inputs(path):
    path = Path(path)
    required = ("top_token_ids.npy", "top_logits.npy", "token_ids.npy", "metadata.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {', '.join(missing)}. "
            "top_token_ids.npy is required to identify the top-1 vocabulary token at each layer; "
            "rerun export_logit_lens.py to regenerate the complete NPY output.")
    top_ids = np.load(path / "top_token_ids.npy", allow_pickle=False)
    top_logits = np.load(path / "top_logits.npy", allow_pickle=False)
    token_ids = np.load(path / "token_ids.npy", allow_pickle=False)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if top_ids.ndim != 4 or top_ids.shape[0] != 2 or top_logits.shape != top_ids.shape:
        raise ValueError(f"Expected top_ids/top_logits shape [2, layer, token, rank], got {top_ids.shape}/{top_logits.shape}")
    if token_ids.shape != (top_ids.shape[2],):
        raise ValueError(f"token_ids shape {token_ids.shape} does not match token axis {top_ids.shape[2]}")
    return path, top_ids, top_logits, token_ids, metadata


def jaccard_rows(left, right):
    values = []
    for a, b in zip(left, right):
        sa, sb = set(int(x) for x in a), set(int(x) for x in b)
        values.append(len(sa & sb) / max(len(sa | sb), 1))
    return np.asarray(values, dtype=np.float64)


def trajectory_metrics(top_ids, top_logits, token_ids, top_k):
    # Shape after slicing: [model, layer, token, rank].
    ids = top_ids[:, :, :, :top_k]
    logits = top_logits[:, :, :, :top_k]
    top1 = ids[:, :, :, 0]  # [model, layer, token]
    models, layers, tokens = top1.shape
    layer_index = np.arange(layers)

    cross_agreement = np.mean(top1[0] == top1[1], axis=1)
    cross_topk_jaccard = np.asarray([
        np.mean(jaccard_rows(ids[0, layer], ids[1, layer])) for layer in range(layers)
    ])
    within_stability = np.full((models, layers), np.nan)
    within_topk_stability = np.full((models, layers), np.nan)
    for model in range(models):
        for layer in range(1, layers):
            within_stability[model, layer] = np.mean(top1[model, layer - 1] == top1[model, layer])
            within_topk_stability[model, layer] = np.mean(
                jaccard_rows(ids[model, layer - 1], ids[model, layer]))

    first_divergence = np.full(tokens, -1, dtype=np.int32)
    for token in range(tokens):
        differing = np.flatnonzero(top1[0, :, token] != top1[1, :, token])
        if differing.size:
            first_divergence[token] = int(differing[0])
    final_flip = top1[:, -2, :] != top1[:, -1, :] if layers >= 2 else np.zeros((models, tokens), bool)
    prefinal_agree = top1[0, -2] == top1[1, -2] if layers >= 2 else np.zeros(tokens, bool)
    final_agreement = top1[0, -1] == top1[1, -1]
    final_new_divergence = prefinal_agree & ~final_agreement if layers >= 2 else np.zeros(tokens, bool)

    # Same-token logit deltas for shared vocabulary IDs in the saved top-k lists.
    final_shared_delta = np.full((layers, tokens), np.nan, dtype=np.float64)
    for layer in range(layers):
        for token in range(tokens):
            a = {int(t): float(v) for t, v in zip(ids[0, layer, token], logits[0, layer, token])}
            b = {int(t): float(v) for t, v in zip(ids[1, layer, token], logits[1, layer, token])}
            common = set(a) & set(b)
            if common:
                final_shared_delta[layer, token] = np.mean([b[t] - a[t] for t in common])

    rows = []
    for token in range(tokens):
        row = {"token_position": token, "input_token_id": int(token_ids[token]),
               "first_cross_model_top1_divergence_layer": int(first_divergence[token]),
               "base_final_layer_flip": int(final_flip[0, token]),
               "k0_final_layer_flip": int(final_flip[1, token]),
               "new_cross_model_divergence_at_final": int(final_new_divergence[token]),
               "base_top1_path_changes": int(np.sum(top1[0, 1:, token] != top1[0, :-1, token])),
               "k0_top1_path_changes": int(np.sum(top1[1, 1:, token] != top1[1, :-1, token])),
               "base_final_top1": int(top1[0, -1, token]), "k0_final_top1": int(top1[1, -1, token])}
        for layer in range(layers):
            row[f"base_top1_layer_{layer}"] = int(top1[0, layer, token])
            row[f"k0_top1_layer_{layer}"] = int(top1[1, layer, token])
        rows.append(row)

    summary = []
    for layer in range(layers):
        summary.append({
            "layer": layer,
            "base_k0_top1_agreement": float(cross_agreement[layer]),
            "base_k0_top_k_jaccard": float(cross_topk_jaccard[layer]),
            "base_top1_adjacent_stability": float(within_stability[0, layer]) if layer else float("nan"),
            "k0_top1_adjacent_stability": float(within_stability[1, layer]) if layer else float("nan"),
            "top1_stability_gap_k0_minus_base": float(within_stability[1, layer] - within_stability[0, layer]) if layer else float("nan"),
            "base_top_k_adjacent_jaccard": float(within_topk_stability[0, layer]) if layer else float("nan"),
            "k0_top_k_adjacent_jaccard": float(within_topk_stability[1, layer]) if layer else float("nan"),
            "top_k_stability_gap_k0_minus_base": float(within_topk_stability[1, layer] - within_topk_stability[0, layer]) if layer else float("nan"),
            "shared_logit_delta_mean": float(np.nanmean(final_shared_delta[layer])),
            "shared_logit_delta_abs_mean": float(np.nanmean(np.abs(final_shared_delta[layer]))),
            "first_divergence_fraction": float(np.mean(first_divergence == layer)),
            "cumulative_divergence_fraction": float(np.mean((first_divergence >= 0) & (first_divergence <= layer))),
            "final_new_divergence_fraction": float(np.mean(final_new_divergence)) if layer == layers - 1 else float("nan"),
            "tokens": tokens,
        })
    arrays = {
        "layer": layer_index,
        "top1": top1,
        "cross_model_top1_agreement": cross_agreement,
        "cross_model_topk_jaccard": cross_topk_jaccard,
        "within_model_top1_stability": within_stability,
        "within_model_topk_jaccard": within_topk_stability,
        "first_divergence_layer": first_divergence,
        "final_new_divergence": final_new_divergence,
        "shared_logit_delta": final_shared_delta,
    }
    return rows, summary, arrays


def write_outputs(out, rows, summary, arrays, metadata):
    out.mkdir(parents=True, exist_ok=True)
    token_path = out / "top1_token_trajectories.csv"
    with token_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary_path = out / "propagation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    np.savez_compressed(out / "propagation_metrics.npz", **arrays)
    (out / "propagation_metadata.json").write_text(json.dumps({
        "source_metadata": metadata,
        "definitions": {
            "first_divergence": "first layer where base and k0 top-1 vocabulary IDs differ",
            "final_new_divergence": "base/k0 agree at layer L-1 but disagree at final layer L",
            "adjacent_stability": "fraction of token positions whose top-1 ID is unchanged from layer l-1 to l",
            "top_k_jaccard": "Jaccard overlap of saved top-k vocabulary ID sets",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layer = arrays["layer"]
    agreement = arrays["cross_model_top1_agreement"]
    jaccard = arrays["cross_model_topk_jaccard"]
    stability = arrays["within_model_top1_stability"]
    topk_stability = arrays["within_model_topk_jaccard"]
    first = arrays["first_divergence_layer"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(layer, agreement, marker="o", label="top-1 agreement")
    axes[0, 0].plot(layer, jaccard, marker="o", label="top-k Jaccard")
    axes[0, 0].set_title("Base vs watermarked agreement by layer")
    axes[0, 0].set_ylabel("agreement / Jaccard"); axes[0, 0].legend()
    axes[0, 1].plot(layer, stability[0], marker="o", label="base")
    axes[0, 1].plot(layer, stability[1], marker="o", label="k0")
    axes[0, 1].set_title("Top-1 propagation continuity")
    axes[0, 1].set_ylabel("same top-1 as previous layer"); axes[0, 1].legend()
    axes[1, 0].plot(layer, topk_stability[0], marker="o", label="base")
    axes[1, 0].plot(layer, topk_stability[1], marker="o", label="k0")
    axes[1, 0].set_title("Top-k propagation continuity")
    axes[1, 0].set_ylabel("adjacent Jaccard"); axes[1, 0].legend()
    counts = np.bincount(first[first >= 0], minlength=len(layer))
    axes[1, 1].bar(layer, counts / max(len(first), 1), color="#dc2626")
    axes[1, 1].set_title("First base/k0 top-1 divergence layer")
    axes[1, 1].set_ylabel("fraction of token positions")
    for ax in axes.ravel():
        ax.set_xlabel("hidden-state layer (0 = embedding)")
        ax.grid(alpha=.25)
    fig.suptitle("Top-token propagation: gradual divergence vs final-layer jump")
    fig.tight_layout()
    plot_path = out / "propagation_analysis.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(layer, [row["shared_logit_delta_mean"] for row in summary], marker="o", label="shared-token mean logit delta")
    ax.plot(layer, [row["shared_logit_delta_abs_mean"] for row in summary], marker="o", label="absolute shared-token delta")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_title("Same-token projected-logit change by layer")
    ax.set_xlabel("hidden-state layer"); ax.set_ylabel("k0 - base"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    magnitude_path = out / "propagation_logit_shift.png"
    fig.savefig(magnitude_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return token_path, summary_path, out / "propagation_metrics.npz", plot_path, magnitude_path


def main():
    args = parse_args()
    source, top_ids, top_logits, token_ids, metadata = load_inputs(args.input_dir)
    saved_k = top_ids.shape[-1]
    top_k = saved_k if args.top_k is None else min(args.top_k, saved_k)
    if top_k < 1:
        raise ValueError("--top-k must be positive")
    rows, summary, arrays = trajectory_metrics(top_ids, top_logits, token_ids, top_k)
    out = Path(args.output_dir) if args.output_dir else source / "propagation_analysis"
    outputs = write_outputs(out, rows, summary, arrays, metadata)
    print(f"Analyzed {top_ids.shape[2]} token positions across {top_ids.shape[1]} layers using top-k={top_k}.")
    for path in outputs:
        print(f"Saved: {path}")
    print("Final-layer diagnostics:")
    print(f"  top-1 agreement at final layer: {summary[-1]['base_k0_top1_agreement']:.5f}")
    print(f"  top-1 stability gap at final layer: {summary[-1]['top1_stability_gap_k0_minus_base']:.5f}")
    print(f"  new cross-model divergence at final layer: {summary[-1]['final_new_divergence_fraction']:.5f}")


if __name__ == "__main__":
    main()
