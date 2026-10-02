#!/usr/bin/env python
"""Plot saved full-vocabulary KGW logit-lens CSV summaries.

Input is full_vocab_group_summary.csv produced by
analyze_kgw_logit_lens_live.py. This script does not load model weights.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", help="full_vocab_group_summary.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--hist-layers", default="1,31,32",
                   help="Layers to show in the mean/error diagnostic")
    args = p.parse_args()
    path = Path(args.csv_path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV is empty")
    for r in rows:
        for k in ("layer", "green_mean", "red_mean", "green_red_gap", "green_count", "red_count", "all_mean", "all_std"):
            r[k] = float(r[k])
    models = list(dict.fromkeys(r["model"] for r in rows))
    layers = sorted(set(int(r["layer"]) for r in rows))
    out = Path(args.output_dir) if args.output_dir else path.parent / "csv_plots"
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Separate base/watermarked panels, with green/red curves and gap.
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 5), squeeze=False, sharey=True)
    for col, model in enumerate(models):
        ax = axes[0, col]
        rr = sorted((r for r in rows if r["model"] == model), key=lambda x: (x.get("prompt", 0), x["layer"]))
        # Prompt curves are thin; bold line is the across-prompt mean.
        prompts = sorted(set(int(r.get("prompt", 0)) for r in rr))
        for prompt in prompts:
            q = sorted((r for r in rr if int(r.get("prompt", 0)) == prompt), key=lambda x: x["layer"])
            ax.plot([r["layer"] for r in q], [r["green_mean"] for r in q], color="#16a34a", alpha=.18)
            ax.plot([r["layer"] for r in q], [r["red_mean"] for r in q], color="#dc2626", alpha=.18)
        mean_by_layer = []
        for layer in layers:
            q = [r for r in rr if int(r["layer"]) == layer]
            mean_by_layer.append((layer, np.mean([r["green_mean"] for r in q]), np.mean([r["red_mean"] for r in q])))
        x = [q[0] for q in mean_by_layer]
        ax.plot(x, [q[1] for q in mean_by_layer], color="#16a34a", linewidth=2.5, label="green mean")
        ax.plot(x, [q[2] for q in mean_by_layer], color="#dc2626", linewidth=2.5, linestyle="--", label="red mean")
        ax.axvline(max(layers) - 1.5, color="gray", linestyle=":")
        ax.set_title(model); ax.set_xlabel("hidden-state layer"); ax.grid(alpha=.25); ax.legend()
    axes[0, 0].set_ylabel("full-vocabulary projected logit")
    fig.suptitle("KGW green/red logit-lens means: separate model panels")
    fig.tight_layout(); fig.savefig(out / "means_base_k0_separate.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    # Gap and coverage diagnostic.
    fig, ax = plt.subplots(figsize=(11, 5))
    for model in models:
        by = []
        for layer in layers:
            q = [r for r in rows if r["model"] == model and int(r["layer"]) == layer]
            by.append(np.mean([r["green_red_gap"] for r in q]))
        ax.plot(layers, by, marker="o", label=f"{model}: green-red gap")
    ax.axhline(0, color="black", linewidth=.8); ax.grid(alpha=.25)
    ax.set_xlabel("hidden-state layer"); ax.set_ylabel("green mean - red mean"); ax.legend()
    ax.set_title("Raw green-red gap from full-vocabulary CSV")
    fig.tight_layout(); fig.savefig(out / "green_red_gap_by_layer.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    # CSV has only means/std/counts, not individual logits. Show mean ±
    # across-prompt standard deviation for requested layers as a diagnostic,
    # not as a reconstructed histogram.
    selected = [int(x) for x in args.hist_layers.split(",")]
    fig, axes = plt.subplots(len(selected), len(models), figsize=(7 * len(models), 3.5 * len(selected)), squeeze=False, sharey="row")
    for i, layer in enumerate(selected):
        for j, model in enumerate(models):
            ax = axes[i, j]
            q = [r for r in rows if r["model"] == model and int(r["layer"]) == layer]
            if not q:
                ax.set_visible(False); continue
            labels = ["green", "red", "all"]
            means = [np.mean([r["green_mean"] for r in q]), np.mean([r["red_mean"] for r in q]), np.mean([r["all_mean"] for r in q])]
            spread = [np.std([r["green_mean"] for r in q]), np.std([r["red_mean"] for r in q]), np.std([r["all_mean"] for r in q])]
            colors = ["#16a34a", "#dc2626", "#64748b"]
            ax.errorbar(labels, means, yerr=spread, fmt="o", color="#111827", capsize=5)
            ax.scatter(labels, means, color=colors, s=70, zorder=3)
            ax.set_title(f"{model}, layer {layer}: prompt mean ± prompt SD")
            ax.set_ylabel("logit"); ax.grid(alpha=.25)
    fig.suptitle("Available CSV summary (not a token-level histogram)")
    fig.tight_layout(); fig.savefig(out / "selected_layer_mean_spread.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"Saved plots under {out}")


if __name__ == "__main__":
    main()
