#!/usr/bin/env python
"""Analyze KGW green/red logit-lens movement for base vs a watermarked model.

The input is produced by export_logit_lens.py. For every retained token
position, candidate vocabulary IDs are classified using the KGW green list
corresponding to that position. The script then computes, separately for each
prompt and model:

  * mean projected logit for green and red saved candidates;
  * adjacent-layer movement (layer l minus layer l-1);
  * per-token movement distributions for the final transitions.

Because export_logit_lens.py stores only top-k candidates, all statistics are
conditional on candidates present in the saved top-k lists. Missing values are
not treated as zero.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tokenizer", required=True,
                   help="Tokenizer used during export, e.g. meta-llama/Llama-2-7b-hf")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--kgw-gamma", type=float, default=0.25)
    p.add_argument("--kgw-seeding-scheme", choices=("simple_0", "simple_1", "simple_2"), default=None,
                   help="Default is inferred from model_b name: k0/k1/k2 -> simple_0/1/2")
    p.add_argument("--last-transitions", type=int, default=3,
                   help="Number of final layer transitions to include in histograms")
    p.add_argument("--bins", type=int, default=80,
                   help="Number of histogram bins (default: 80; larger gives narrower bars)")
    return p.parse_args()


def load_data(directory):
    directory = Path(directory)
    ids = np.load(directory / "top_token_ids.npy", allow_pickle=False)
    logits = np.load(directory / "top_logits.npy", allow_pickle=False)
    token_ids = np.load(directory / "token_ids.npy", allow_pickle=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if ids.ndim != 4 or ids.shape[0] != 2 or logits.shape != ids.shape:
        raise ValueError(f"Expected arrays [2, layer, token, rank], got {ids.shape}/{logits.shape}")
    if token_ids.shape != (ids.shape[2],):
        raise ValueError(f"token_ids shape {token_ids.shape} does not match token axis {ids.shape[2]}")
    return directory, ids, logits, token_ids, metadata


def infer_scheme(metadata, explicit):
    if explicit:
        return explicit
    source = metadata.get("model_b", {}).get("source", "")
    for variant, scheme in (("-k0-", "simple_0"), ("-k1-", "simple_1"), ("-k2-", "simple_2")):
        if variant in source:
            return scheme
    return "simple_1"


def reconstruct_positions(metadata, tokenizer, saved_token_ids):
    """Reproduce exporter flattening and max_tokens subsampling exactly."""
    prompts = metadata.get("prompts")
    if not prompts:
        raise ValueError("metadata.json has no prompts")
    encoded = tokenizer(prompts, add_special_tokens=True, padding=False, truncation=False)
    flat_ids = []
    prompt_index = []
    position_in_prompt = []
    for prompt_id, ids in enumerate(encoded["input_ids"]):
        for position, token in enumerate(ids):
            flat_ids.append(int(token))
            prompt_index.append(prompt_id)
            position_in_prompt.append(position)
    flat_ids = np.asarray(flat_ids, dtype=np.int64)
    saved = len(saved_token_ids)
    if len(flat_ids) < saved:
        raise ValueError(f"Tokenizer reconstructed {len(flat_ids)} positions but NPY has {saved}")
    indices = np.linspace(0, len(flat_ids) - 1, saved, dtype=np.int64) if len(flat_ids) > saved else np.arange(saved)
    selected_ids = flat_ids[indices]
    if not np.array_equal(selected_ids, saved_token_ids):
        raise ValueError("Tokenizer/prompt alignment differs from export; use the exact tokenizer and export prompts")
    return np.asarray(prompt_index, dtype=np.int64)[indices], np.asarray(position_in_prompt, dtype=np.int64)[indices], encoded


def build_green_sets(encoded, tokenizer, vocab_size, gamma, scheme):
    import torch
    from watermarks.kgw.watermark_processor import WatermarkBase

    watermark = WatermarkBase(vocab=list(range(vocab_size)), gamma=gamma,
                              seeding_scheme=scheme, device="cpu")
    special_ids = set()
    if scheme == "simple_1":
        special_ids = {int(x) for x in (tokenizer.eos_token_id, tokenizer.bos_token_id,
                                         tokenizer.pad_token_id, tokenizer.unk_token_id)
                       if x is not None and 0 <= int(x) < vocab_size}
    result = []
    for ids in encoded["input_ids"]:
        ids = np.asarray(ids, dtype=np.int64)
        for position in range(len(ids)):
            context = ids[:position + 1]
            if len(context) < watermark.context_width:
                result.append(set())
                continue
            if scheme == "simple_1" and int(context[-1]) in special_ids:
                result.append(set())
                continue
            green = watermark._get_greenlist_ids(torch.as_tensor(context, dtype=torch.long))
            green_set = {int(x) for x in green.tolist()} - special_ids
            result.append(green_set)
    return result


def select_masks(all_sets, total_positions, saved_positions):
    indices = np.linspace(0, total_positions - 1, saved_positions, dtype=np.int64) if total_positions > saved_positions else np.arange(saved_positions)
    return [all_sets[int(index)] for index in indices]


def group_arrays(ids, logits, green_sets, top_k):
    layers = ids.shape[1]
    models = ids.shape[0]
    result = {}
    for model in range(models):
        green = np.full((layers, len(green_sets)), np.nan, dtype=np.float64)
        red = np.full_like(green, np.nan)
        for layer in range(layers):
            for position, green_set in enumerate(green_sets):
                row_ids = ids[model, layer, position, :top_k]
                row_values = logits[model, layer, position, :top_k]
                is_green = np.asarray([int(token) in green_set for token in row_ids], dtype=bool)
                if is_green.any():
                    green[layer, position] = float(np.mean(row_values[is_green]))
                if (~is_green).any():
                    red[layer, position] = float(np.mean(row_values[~is_green]))
        result[model] = {"green": green, "red": red}
    return result


def token_movements(ids, logits, green_sets, prompt_index, top_k):
    """Return per-token movement values matched by vocabulary ID across layers.

    Unlike group_arrays(), this does not average all green/red candidates within
    a position first. Each vocabulary token that appears in both adjacent
    layers contributes one movement observation.
    """
    models, layers, positions, _ = ids.shape
    result = {}
    for model in range(models):
        result[model] = {}
        for layer in range(1, layers):
            for group in ("green", "red", "all"):
                result[model, layer, group] = {int(prompt): [] for prompt in sorted(set(prompt_index.tolist()))}
            for position in range(positions):
                previous = {int(t): float(v) for t, v in zip(ids[model, layer - 1, position, :top_k], logits[model, layer - 1, position, :top_k])}
                current = {int(t): float(v) for t, v in zip(ids[model, layer, position, :top_k], logits[model, layer, position, :top_k])}
                for token in set(previous) & set(current):
                    movement = current[token] - previous[token]
                    group = "green" if token in green_sets[position] else "red"
                    prompt = int(prompt_index[position])
                    result[model, layer, group][prompt].append(movement)
                    result[model, layer, "all"][prompt].append(movement)
    return result


def rows_for_csv(grouped, prompt_index, model_names):
    layers, positions = grouped[0]["green"].shape
    rows = []
    for model, model_name in enumerate(model_names):
        for prompt in sorted(set(prompt_index.tolist())):
            selected = np.flatnonzero(prompt_index == prompt)
            for layer in range(layers):
                g = grouped[model]["green"][layer, selected]
                r = grouped[model]["red"][layer, selected]
                g_mean = float(np.nanmean(g)) if np.isfinite(g).any() else float("nan")
                r_mean = float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")
                if layer == 0:
                    g_move = r_move = float("nan")
                else:
                    gp = grouped[model]["green"][layer - 1, selected]
                    rp = grouped[model]["red"][layer - 1, selected]
                    g_move = float(np.nanmean(g - gp)) if np.isfinite(g - gp).any() else float("nan")
                    r_move = float(np.nanmean(r - rp)) if np.isfinite(r - rp).any() else float("nan")
                rows.append({"model": model_name, "prompt": int(prompt), "layer": layer,
                             "green_mean_logit": g_mean, "red_mean_logit": r_mean,
                             "green_minus_red": g_mean - r_mean,
                             "green_movement": g_move, "red_movement": r_move,
                             "green_red_movement_gap": g_move - r_move if np.isfinite(g_move) and np.isfinite(r_move) else float("nan"),
                             "green_count": int(np.isfinite(grouped[model]["green"][layer, selected]).sum()),
                             "red_count": int(np.isfinite(grouped[model]["red"][layer, selected]).sum())})
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def raw_gap_by_prompt(grouped, prompt_index):
    """Return [model, prompt, layer] green-minus-red mean-logit gaps."""
    prompts = sorted(set(prompt_index.tolist()))
    layers = grouped[0]["green"].shape[0]
    gaps = np.full((2, len(prompts), layers), np.nan, dtype=np.float64)
    for model in range(2):
        for prompt_row, prompt in enumerate(prompts):
            selected = np.flatnonzero(prompt_index == prompt)
            for layer in range(layers):
                green = grouped[model]["green"][layer, selected]
                red = grouped[model]["red"][layer, selected]
                if np.isfinite(green).any() and np.isfinite(red).any():
                    gaps[model, prompt_row, layer] = np.nanmean(green) - np.nanmean(red)
    return prompts, gaps


def make_plots(out, grouped, movements, prompt_index, model_names, last_transitions, bins, raw_gaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = grouped[0]["green"].shape[0]
    prompts = sorted(set(prompt_index.tolist()))
    # Plot the mean movement of all matched vocabulary tokens in each KGW
    # group. Color encodes KGW class; line style encodes model.
    fig, axes = plt.subplots(len(prompts), 1, figsize=(12, max(5, 3.0 * len(prompts))), squeeze=False, sharex=True)
    for row, prompt in enumerate(prompts):
        ax = axes[row, 0]
        for model, name in enumerate(model_names):
            for group, color in (("green", "#16a34a"), ("red", "#dc2626")):
                values = np.full(layers, np.nan)
                for layer in range(1, layers):
                    sample = np.asarray(movements[model, layer, group][prompt], dtype=float)
                    if sample.size:
                        values[layer] = np.mean(sample)
                style = "-" if model == 0 else "--"
                ax.plot(np.arange(layers), values, color=color, linestyle=style,
                        linewidth=2.0, label=f"{name} {group}")
        ax.axhline(0, color="black", linewidth=.8)
        ax.axvline(layers - 1.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Prompt {prompt}: mean movement of matched tokens")
        ax.set_ylabel("Δ logit")
        ax.grid(alpha=.25)
        ax.legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("layer transition (value at l = l - 1)")
    fig.suptitle("KGW green/red mean logit-lens movement: base vs watermarked")
    fig.tight_layout()
    movement_path = out / "kgw_group_movement_by_prompt.png"
    fig.savefig(movement_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # For each requested late transition, pool all matched token movements
    # across prompts and positions. Base should be near a central distribution;
    # a KGW split in the watermarked model can produce two peaks.
    transition_layers = list(range(max(1, layers - last_transitions), layers))
    fig, axes = plt.subplots(len(transition_layers), 2, figsize=(15, max(4, 3.4 * len(transition_layers))), squeeze=False, sharex="row")
    for row, layer in enumerate(transition_layers):
        for model, name in enumerate(model_names):
            ax = axes[row, model]
            sample = np.asarray(movements[model, layer, "all"][0] + movements[model, layer, "all"][1], dtype=float)
            sample = sample[np.isfinite(sample)]
            if sample.size:
                # Use a shared fine-grained bin range within each transition,
                # so base and watermarked panels have comparable narrow bars.
                ax.hist(sample, bins=bins, color="#64748b" if model == 0 else "#f97316",
                        alpha=.72, edgecolor="white", linewidth=.35)
            ax.axvline(0, color="black", linewidth=.8)
            ax.set_title(f"{name}: layer {layer - 1} → {layer}")
            ax.set_xlabel("logit movement")
            ax.set_ylabel("count")
            ax.grid(alpha=.2)
    fig.suptitle("All matched-token logit-lens movement distributions")
    fig.tight_layout()
    hist_path = out / "kgw_all_token_movement_histograms.png"
    fig.savefig(hist_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # Optional diagnostic: overlay green/red distributions only for the final
    # transition, with base and watermarked panels side by side.
    final = layers - 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for model, name in enumerate(model_names):
        ax = axes[model]
        for group, color in (("green", "#16a34a"), ("red", "#dc2626")):
            sample = np.asarray(movements[model, final, group][0] + movements[model, final, group][1], dtype=float)
            sample = sample[np.isfinite(sample)]
            if sample.size:
                ax.hist(sample, bins=bins, alpha=.52, color=color,
                        edgecolor="white", linewidth=.35,
                        label=f"{group} (n={sample.size})")
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(f"{name}: final transition {final - 1} → {final}")
        ax.set_xlabel("logit movement"); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel("count")
    fig.suptitle("Final transition split by KGW red/green class")
    fig.tight_layout()
    gap_path = out / "kgw_final_green_red_movement_gap.png"
    fig.savefig(gap_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    # Detailed diagnostic: raw green-red gap and its layer-to-layer change.
    prompts, gaps = raw_gaps
    fig, axes = plt.subplots(len(prompts), 2, figsize=(15, max(5, 3.2 * len(prompts))),
                             squeeze=False, sharex=True)
    for row, prompt in enumerate(prompts):
        for model, name in enumerate(model_names):
            ax = axes[row, model]
            ax.plot(np.arange(layers), gaps[model, row], color="#16a34a", linewidth=1.8,
                    label="green mean − red mean")
            if layers > 1:
                ax.plot(np.arange(1, layers), np.diff(gaps[model, row]), color="#7c3aed",
                        linestyle="--", linewidth=1.2, label="change in gap")
            ax.axhline(0, color="black", linewidth=.8)
            ax.axvline(layers - 1.5, color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"Prompt {prompt}: {name}")
            ax.set_ylabel("raw gap / gap change")
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("hidden-state layer")
    axes[-1, 1].set_xlabel("hidden-state layer")
    fig.suptitle("When does the KGW green-red logit gap appear?")
    fig.tight_layout()
    raw_path = out / "kgw_raw_green_red_gap_by_prompt.png"
    fig.savefig(raw_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # Direct onset view: watermarked raw gap, base raw gap, and extra gap.
    fig, ax = plt.subplots(figsize=(12, 5))
    base_mean = np.nanmean(gaps[0], axis=0)
    water_mean = np.nanmean(gaps[1], axis=0)
    extra = water_mean - base_mean
    ax.plot(np.arange(layers), base_mean, color="#2563eb", label="base raw green-red gap")
    ax.plot(np.arange(layers), water_mean, color="#f97316", label="watermarked raw green-red gap")
    ax.plot(np.arange(layers), extra, color="#7c3aed", linestyle="--", label="watermarked minus base gap")
    ax.axhline(0, color="black", linewidth=.8)
    ax.axvline(layers - 1.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Average KGW green-red gap and watermark-specific excess")
    ax.set_xlabel("hidden-state layer"); ax.set_ylabel("mean projected-logit gap")
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    onset_path = out / "kgw_gap_onset_summary.png"
    fig.savefig(onset_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return movement_path, hist_path, gap_path, raw_path, onset_path


def main():
    args = parse_args()
    source, ids, logits, token_ids, metadata = load_data(args.input_dir)
    top_k = ids.shape[-1] if args.top_k is None else min(args.top_k, ids.shape[-1])
    if top_k < 1:
        raise ValueError("--top-k must be positive")
    scheme = infer_scheme(metadata, args.kgw_seeding_scheme)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=args.hf_token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_index, _, encoded = reconstruct_positions(metadata, tokenizer, token_ids)
    all_sets = build_green_sets(encoded, tokenizer, int(metadata["model_a"]["vocab_size"]), args.kgw_gamma, scheme)
    green_sets = select_masks(all_sets, len(all_sets), len(token_ids))
    grouped = group_arrays(ids, logits, green_sets, top_k)
    movements = token_movements(ids, logits, green_sets, prompt_index, top_k)
    model_b_source = metadata.get("model_b", {}).get("source", "k1")
    model_b_name = "k1" if "-k1-" in model_b_source else ("k0" if "-k0-" in model_b_source else "watermarked")
    model_names = ("base", model_b_name)
    rows = rows_for_csv(grouped, prompt_index, model_names)
    out = Path(args.output_dir) if args.output_dir else source / "kgw_group_analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "kgw_group_layer_summary.csv", rows)
    raw_gaps = raw_gap_by_prompt(grouped, prompt_index)
    plot_paths = make_plots(out, grouped, movements, prompt_index, model_names,
                            args.last_transitions, args.bins, raw_gaps)
    np.savez_compressed(out / "kgw_group_metrics.npz",
                        green=np.stack([grouped[0]["green"], grouped[1]["green"]]),
                        red=np.stack([grouped[0]["red"], grouped[1]["red"]]),
                        prompt_index=prompt_index,
                        raw_green_red_gap=raw_gaps[1],
                        raw_green_red_gap_base=raw_gaps[1][0],
                        raw_green_red_gap_watermarked=raw_gaps[1][1])
    (out / "kgw_group_metadata.json").write_text(json.dumps({
        "source_metadata": metadata, "model_names": model_names,
        "kgw_gamma": args.kgw_gamma, "kgw_seeding_scheme": scheme, "top_k": top_k,
        "definitions": {
            "movement": "mean projected-logit at layer l minus layer l-1",
            "green_red": "candidate vocabulary IDs classified by KGW green list for each retained context",
            "coverage": "only saved top-k candidates; missing candidates are NaN",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Analyzed {len(token_ids)} retained positions, {len(set(prompt_index.tolist()))} prompts, scheme={scheme}, top-k={top_k}")
    for path in plot_paths:
        print(f"Saved plot: {path}")
    print(f"Saved table: {out / 'kgw_group_layer_summary.csv'}")


if __name__ == "__main__":
    main()
