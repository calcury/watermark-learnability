#!/usr/bin/env python
"""Visualize top-token trajectories and late-layer logit jumps.

This uses the saved top-k logit-lens arrays without loading model weights.
It produces:
  * a CSV table of layer-by-layer top-1 tokens for selected positions;
  * a CSV table tracking final-layer candidate tokens across layers;
  * a heatmap of top-1 token IDs by layer and position;
  * plots of top-1 logits and late-layer fixed-token logit curves.

Important: only saved top-k logits are available. A fixed token is NaN at
layers where it was not among the saved top-k candidates.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# When executed as ``python analysis/...py``, Python puts ``analysis/`` first
# on sys.path. Add the repository root so the local ``watermarks`` namespace
# package is importable from Colab and other working directories.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--positions", default=None,
                   help="Comma-separated retained token positions; default selects representative positions")
    p.add_argument("--num-examples", type=int, default=16,
                   help="Number of representative positions when --positions is omitted")
    p.add_argument("--top-k", type=int, default=None,
                   help="Use first K saved ranks; default uses all saved ranks")
    p.add_argument("--candidate-count", type=int, default=12,
                   help="Maximum final-layer candidate curves per position")
    p.add_argument("--tokenizer", default=None,
                   help="Optional tokenizer path/Hub ID for decoded token strings")
    p.add_argument("--kgw-gamma", type=float, default=0.25)
    p.add_argument("--kgw-seeding-scheme", choices=("simple_0", "simple_1", "simple_2"), default="simple_0")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def load_data(directory):
    directory = Path(directory)
    top_ids = np.load(directory / "top_token_ids.npy", allow_pickle=False)
    top_logits = np.load(directory / "top_logits.npy", allow_pickle=False)
    token_ids = np.load(directory / "token_ids.npy", allow_pickle=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if top_ids.ndim != 4 or top_ids.shape[0] != 2 or top_logits.shape != top_ids.shape:
        raise ValueError(f"Expected [2, layer, token, rank] arrays, got {top_ids.shape}/{top_logits.shape}")
    if token_ids.shape != (top_ids.shape[2],):
        raise ValueError(f"token_ids shape {token_ids.shape} does not match {top_ids.shape[2]} token positions")
    return directory, top_ids, top_logits, token_ids, metadata


def choose_positions(tokens, positions, num_examples):
    if positions:
        selected = [int(x.strip()) for x in positions.split(",") if x.strip()]
        if not selected or min(selected) < 0 or max(selected) >= tokens:
            raise ValueError(f"--positions must be between 0 and {tokens - 1}")
        return selected
    count = min(max(1, num_examples), tokens)
    return np.linspace(0, tokens - 1, count, dtype=int).tolist()


def make_decoder(tokenizer_path, metadata):
    source = tokenizer_path or metadata.get("model_a", {}).get("source")
    if not source:
        return lambda token_id: f"<id:{token_id}>"
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
        return lambda token_id: tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False).replace("\n", "\\n")
    except Exception as exc:
        print(f"Warning: tokenizer unavailable ({exc}); using token IDs")
        return lambda token_id: f"<id:{token_id}>"


def lookup_token(ids, values, token_id):
    hits = np.flatnonzero(ids == token_id)
    if hits.size == 0:
        return np.nan
    return float(values[hits[0]])


def build_kgw_green_mask(metadata, token_ids, tokenizer_path, hf_token, gamma, scheme):
    """Recreate the KGW green list for each retained valid token position."""
    try:
        from transformers import AutoTokenizer
        from watermarks.kgw.watermark_processor import WatermarkBase
    except ImportError as exc:
        raise RuntimeError("KGW mask coloring requires transformers and the repository watermarks package") from exc
    source = tokenizer_path or metadata.get("model_a", {}).get("source")
    if not source:
        raise ValueError("A tokenizer/model source is required to build KGW masks")
    tokenizer = AutoTokenizer.from_pretrained(source, token=hf_token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = metadata.get("prompts")
    if not prompts:
        raise ValueError("metadata.json does not contain prompts; cannot reconstruct KGW context")
    encoded = tokenizer(prompts, add_special_tokens=True, padding=False, truncation=False)
    base = WatermarkBase(vocab=list(range(metadata["model_a"]["vocab_size"])),
                         gamma=gamma, seeding_scheme=scheme, device="cpu")
    import torch
    flat_green = []
    for ids in encoded["input_ids"]:
        ids = np.asarray(ids, dtype=np.int64)
        for position in range(len(ids)):
            context = ids[:position + 1]
            if len(context) < base.context_width:
                flat_green.append(set())
                continue
            green_ids = set(int(x) for x in base._get_greenlist_ids(
                torch.as_tensor(context, dtype=torch.long)).tolist())
            flat_green.append(green_ids)
    # export_logit_lens flattens all valid positions and, if necessary, keeps
    # evenly spaced indices. Reproduce that selection for mask alignment.
    total = len(flat_green)
    saved = len(token_ids)
    if total < saved:
        raise ValueError(f"Reconstructed {total} positions but NPY contains {saved}")
    indices = np.linspace(0, total - 1, saved, dtype=np.int64) if total > saved else np.arange(total)
    selected = [flat_green[index] for index in indices]
    return selected


def build_tables(top_ids, top_logits, token_ids, positions, top_k, candidate_count, decode, green_sets):
    models = ("base", "k0")
    layers = top_ids.shape[1]
    top1_rows, fixed_rows = [], []
    # Track each selected position's per-layer top-1 output.
    for position in positions:
        for layer in range(layers):
            for model_index, model in enumerate(models):
                token = int(top_ids[model_index, layer, position, 0])
                value = float(top_logits[model_index, layer, position, 0])
                top1_rows.append({
                    "position": position, "input_token_id": int(token_ids[position]),
                    "model": model, "layer": layer, "top1_token_id": token,
                    "top1_token": decode(token), "top1_logit": value,
                })
        # Use final-layer top-1 candidates so the plot asks when the final
        # decision became available in the preceding representation layers.
        candidates = set(int(x) for model in range(2) for x in top_ids[model, -1, position, :top_k])
        for candidate in sorted(candidates)[:candidate_count]:
            row = {"position": position, "input_token_id": int(token_ids[position]),
                   "candidate_token_id": candidate, "candidate_token": decode(candidate),
                   "kgw_class": "green" if candidate in green_sets[position] else "red",
                   "kgw_is_green": int(candidate in green_sets[position])}
            for model_index, model in enumerate(models):
                for layer in range(layers):
                    row[f"{model}_layer_{layer}_logit"] = lookup_token(
                        top_ids[model_index, layer, position, :top_k],
                        top_logits[model_index, layer, position, :top_k], candidate)
            fixed_rows.append(row)
    return top1_rows, fixed_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_outputs(out, top_ids, top_logits, token_ids, positions, top_k, candidate_count, top1_rows, fixed_rows, decode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ("base", "k0")
    layers = np.arange(top_ids.shape[1])
    # Figure 1: selected examples' top-1 logit and top-token ID heatmap.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for model_index, model in enumerate(models):
        ax = axes[0, model_index]
        for position in positions:
            ax.plot(layers, top_logits[model_index, :, position, 0], marker="o", alpha=.75,
                    label=f"pos {position}")
        ax.set_title(f"{model}: top-1 logit by layer")
        ax.set_xlabel("layer"); ax.set_ylabel("top-1 projected logit")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    for model_index, model in enumerate(models):
        ax = axes[1, model_index]
        image = top_ids[model_index, :, positions, 0].T
        im = ax.imshow(image, aspect="auto", interpolation="nearest", cmap="tab20")
        ax.set_title(f"{model}: top-1 token ID trajectory")
        ax.set_xlabel("layer"); ax.set_ylabel("selected position")
        ax.set_yticks(np.arange(len(positions))); ax.set_yticklabels([str(x) for x in positions])
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Layerwise top-token propagation")
    fig.tight_layout()
    trajectory_path = out / "top_token_trajectory_overview.png"
    fig.savefig(trajectory_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # Figure 2: fixed final-layer candidates. Missing values are intentional.
    fig, axes = plt.subplots(len(positions), 2, figsize=(16, max(5, 3.2 * len(positions))), squeeze=False)
    for row_index, position in enumerate(positions):
        candidates = sorted({r["candidate_token_id"] for r in fixed_rows if r["position"] == position})
        for model_index, model in enumerate(models):
            ax = axes[row_index, model_index]
            for candidate in candidates[:candidate_count]:
                matching = next(r for r in fixed_rows if r["position"] == position and r["candidate_token_id"] == candidate)
                values = np.array([matching.get(f"{model}_layer_{layer}_logit", np.nan) for layer in layers], dtype=float)
                # Neutral gray for the trajectory; color only the final
                # transition according to this candidate's KGW class.
                ax.plot(layers, values, marker=".", color="#9ca3af", alpha=.55, linewidth=1.0)
                if np.isfinite(values[-2]) and np.isfinite(values[-1]):
                    final_color = "#16a34a" if matching["kgw_is_green"] else "#dc2626"
                    ax.plot(layers[-2:], values[-2:], color=final_color, linewidth=2.8, alpha=.95)
                    ax.scatter(layers[-1], values[-1], color=final_color, s=20, zorder=4)
            ax.axvline(layers[-2], color="black", linestyle="--", linewidth=.8, alpha=.6)
            ax.set_title(f"pos {position}, {model}: final-candidate logits")
            ax.set_xlabel("layer"); ax.set_ylabel("projected logit")
            # Deliberately omit token-ID legends: they obscure the many curves.
            ax.grid(alpha=.25)
    fig.suptitle("Do final top-token decisions emerge gradually or jump at the end?", y=.995)
    fig.tight_layout()
    fixed_path = out / "final_candidate_logit_trajectories.png"
    fig.savefig(fixed_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    return trajectory_path, fixed_path


def main():
    args = parse_args()
    source, top_ids, top_logits, token_ids, metadata = load_data(args.input_dir)
    top_k = top_ids.shape[-1] if args.top_k is None else min(args.top_k, top_ids.shape[-1])
    if top_k < 1:
        raise ValueError("--top-k must be positive")
    positions = choose_positions(top_ids.shape[2], args.positions, args.num_examples)
    decode = make_decoder(args.tokenizer, metadata)
    green_sets = build_kgw_green_mask(metadata, token_ids, args.tokenizer, args.hf_token,
                                      args.kgw_gamma, args.kgw_seeding_scheme)
    top1_rows, fixed_rows = build_tables(top_ids, top_logits, token_ids, positions, top_k,
                                         args.candidate_count, decode, green_sets)
    out = Path(args.output_dir) if args.output_dir else source / "trajectory_visualization"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "selected_top1_tokens_by_layer.csv", top1_rows)
    write_csv(out / "selected_final_candidate_logits_by_layer.csv", fixed_rows)
    paths = plot_outputs(out, top_ids, top_logits, token_ids, positions, top_k,
                         args.candidate_count, top1_rows, fixed_rows, decode)
    (out / "visualization_metadata.json").write_text(json.dumps({
        "source_metadata": metadata, "selected_positions": positions, "top_k": top_k,
        "limitation": "fixed-token curves are NaN when the token is outside saved top-k at a layer",
        "interpretation": "each layer is a projection of the same hidden position, not a generated decoding timestep",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected token positions: {positions}")
    print(f"Saved table: {out / 'selected_top1_tokens_by_layer.csv'}")
    print(f"Saved fixed-token table: {out / 'selected_final_candidate_logits_by_layer.csv'}")
    for path in paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
