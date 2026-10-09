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


def save_results(prefix, args, manifest, rows):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"input": args.input, "metadata": manifest.get("metadata", {}),
               "test_size": args.test_size, "seed": args.seed, "rows": rows}
    json_path = prefix.with_suffix(".json")
    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_json.replace(json_path)
    csv_path = prefix.with_suffix(".csv")
    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tmp_csv.replace(csv_path)
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
        pass


def main():
    args = parse_args()
    manifest, parts = load_data(args.input)
    n_layers = len(parts[0]["embeddings"])
    rng = np.random.default_rng(args.seed)
    prefix = Path(args.output_prefix)
    rows = []
    existing_json = prefix.with_suffix(".json")
    if existing_json.exists():
        try:
            previous = json.loads(existing_json.read_text(encoding="utf-8"))
            compatible = (previous.get("input") == args.input and
                          previous.get("seed") == args.seed and
                          float(previous.get("test_size", args.test_size)) == args.test_size)
            if compatible:
                rows = sorted(previous.get("rows", []), key=lambda r: int(r["layer"]))
                print(f"resuming: found {len(rows)} completed layer(s)", flush=True)
            else:
                print("existing result parameters differ; starting from layer 0", flush=True)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"could not read existing results ({exc}); starting over", flush=True)
    completed = {int(row["layer"]) for row in rows}
    for layer in range(n_layers):
        if layer in completed:
            print(f"layer={layer:>2} already saved; skipping", flush=True)
            continue
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
        rows = [r for r in rows if int(r["layer"]) != layer]
        rows.append(row)
        rows.sort(key=lambda r: int(r["layer"]))
        save_results(prefix, args, manifest, rows)
        print(f"saved checkpoint through layer {layer}", flush=True)
    if rows:
        save_results(prefix, args, manifest, rows)
    print(f"saved incremental results: {prefix.with_suffix('.csv')} and {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
