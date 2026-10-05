#!/usr/bin/env python3
"""Fit a layer-wise linear probe from normal or chunked extractor output."""
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
    p.add_argument("--input", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--max-points", type=int, default=0)
    return p.parse_args()


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "embeddings" in data:
        return data, [data]
    if data.get("format") != "kgw_probe_chunks_v1":
        raise ValueError(f"Unsupported input format: {path}")
    parts = []
    for part_path in data["chunks"]:
        parts.append(torch.load(part_path, map_location="cpu", weights_only=False))
    return data, parts


def main():
    args = parse_args()
    manifest, parts = load_data(args.input)
    n_layers = len(parts[0]["embeddings"])
    rng = np.random.default_rng(args.seed)
    rows = []
    for layer in range(n_layers):
        x = torch.cat([p["embeddings"][layer] for p in parts], dim=0).numpy().astype(np.float32, copy=False)
        y = torch.cat([p["labels"] for p in parts], dim=0).numpy().astype(np.int64, copy=False)
        if args.max_points and len(y) > args.max_points:
            keep = rng.choice(len(y), args.max_points, replace=False)
            x, y = x[keep], y[keep]
        if len(np.unique(y)) != 2:
            raise ValueError(f"Layer {layer} input contains only one class")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.seed, stratify=y)
        clf = LogisticRegression(C=args.c, max_iter=args.max_iter, solver="lbfgs", n_jobs=1)
        clf.fit(x_train, y_train)
        train_pred = clf.predict(x_train)
        test_pred = clf.predict(x_test)
        correct = int(np.sum(test_pred == y_test))
        n_test = int(len(y_test))
        acc = float(accuracy_score(y_test, test_pred))
        p_value = float(binom.sf(correct - 1, n_test, 0.5))
        row = {"layer": layer, "n_train": len(y_train), "n_test": n_test,
               "train_accuracy": float(accuracy_score(y_train, train_pred)),
               "accuracy": acc, "correct": correct, "chance": 0.5,
               "p_value": p_value, "log10_p_value": float(np.log10(max(p_value, 1e-300)))}
        rows.append(row)
        print(f"layer={layer:>2} train={row['train_accuracy']:.4f} test={acc:.4f} p={p_value:.4g}", flush=True)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    json_path = prefix.with_suffix(".json")
    metadata = manifest.get("metadata", {})
    json_path.write_text(json.dumps({"input": args.input, "metadata": metadata,
                                     "test_size": args.test_size, "seed": args.seed, "rows": rows}, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        layers = [r["layer"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(layers, [r["accuracy"] for r in rows], marker="o", label="test")
        axes[0].plot(layers, [r["train_accuracy"] for r in rows], marker=".", alpha=.6, label="train")
        axes[0].axhline(.5, color="gray", ls="--"); axes[0].set_ylim(0, 1); axes[0].legend()
        axes[0].set(xlabel="Layer (0 = embedding)", ylabel="Accuracy", title="KGW linear probe")
        axes[1].plot(layers, [r["p_value"] for r in rows], marker="o")
        axes[1].axhline(.05, color="gray", ls="--"); axes[1].set_yscale("log")
        axes[1].set(xlabel="Layer", ylabel="One-sided binomial p-value", title="Test accuracy significance")
        fig.tight_layout(); fig.savefig(prefix.with_suffix(".png"), dpi=180); plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; CSV and JSON were still written")
    print(f"saved {csv_path}\nsaved {json_path}")


if __name__ == "__main__":
    main()
