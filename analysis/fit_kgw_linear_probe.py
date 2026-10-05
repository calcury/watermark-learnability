#!/usr/bin/env python3
"""Fit a linear probe to saved KGW hidden-state embeddings, layer by layer.

The reported p-value is the one-sided exact binomial test of test accuracy
against 0.5 (the saved dataset is normally balanced by the extractor).

Example:
  python analysis/fit_kgw_linear_probe.py \
    --input analysis/kgw_probe_k0_d2.pt \
    --output-prefix analysis/kgw_probe_k0_d2
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help=".pt file from extract_kgw_probe_embeddings.py")
    p.add_argument("--output-prefix", required=True, help="Prefix for .csv, .json, and .png outputs")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--c", type=float, default=1.0, help="Logistic-regression inverse regularization")
    p.add_argument("--max-points", type=int, default=0,
                   help="Optional random cap per layer to reduce fitting time; 0 means all")
    return p.parse_args()


def main():
    args = parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]
    y = data["labels"].numpy().astype(np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("The input must contain both green and red labels")
    rng = np.random.default_rng(args.seed)
    rows = []
    for layer, tensor in enumerate(embeddings):
        x = tensor.numpy().astype(np.float32, copy=False)
        if args.max_points and len(y) > args.max_points:
            keep = rng.choice(len(y), args.max_points, replace=False)
            x_layer, y_layer = x[keep], y[keep]
        else:
            x_layer, y_layer = x, y
        x_train, x_test, y_train, y_test = train_test_split(
            x_layer, y_layer, test_size=args.test_size, random_state=args.seed,
            stratify=y_layer)
        clf = LogisticRegression(C=args.c, max_iter=args.max_iter,
                                 solver="lbfgs", n_jobs=1)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        correct = int(np.sum(pred == y_test))
        n_test = int(len(y_test))
        accuracy = float(accuracy_score(y_test, pred))
        p_value = float(binom.sf(correct - 1, n_test, 0.5))
        rows.append({"layer": layer, "n_train": len(y_train), "n_test": n_test,
                     "accuracy": accuracy, "correct": correct,
                     "chance": 0.5, "p_value": p_value,
                     "log10_p_value": float(np.log10(max(p_value, 1e-300)))})
        print(f"layer={layer:>2} accuracy={accuracy:.4f} p={p_value:.4g}")
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps({"input": args.input, "metadata": data.get("metadata", {}),
                                     "test_size": args.test_size, "seed": args.seed,
                                     "rows": rows}, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        layers = [r["layer"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        pvals = [r["p_value"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(layers, acc, marker="o")
        axes[0].axhline(0.5, color="gray", linestyle="--", label="chance")
        axes[0].set(xlabel="Hidden-state layer (0 = embeddings)", ylabel="Test accuracy",
                    title="KGW green/red linear probe")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[1].plot(layers, pvals, marker="o")
        axes[1].axhline(0.05, color="gray", linestyle="--", label="p=0.05")
        axes[1].set_yscale("log")
        axes[1].set(xlabel="Hidden-state layer", ylabel="One-sided binomial p-value",
                    title="Accuracy significance vs 0.5")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(prefix.with_suffix(".png"), dpi=180)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; CSV and JSON were still written")
    print(f"saved {csv_path}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
