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
    p.add_argument("--bins", type=int, default=30)
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


def make_plots(out, grouped, prompt_index, model_names, last_transitions, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = grouped[0]["green"].shape[0]
    prompts = sorted(set(prompt_index.tolist()))
    colors = {"base": "#2563eb", "k1": "#f97316", "k0": "#f97316"}
    # Prompt-specific mean movement curves: one subplot per prompt.
    fig, axes = plt.subplots(len(prompts), 1, figsize=(12, max(5, 3.0 * len(prompts))), squeeze=False, sharex=True)
    for row, prompt in enumerate(prompts):
        ax = axes[row, 0]
        selected = np.flatnonzero(prompt_index == prompt)
        for model, name in enumerate(model_names):
            color = colors.get(name, ["#2563eb", "#f97316"][model])
            for group, linestyle in (("green", "-"), ("red", "--")):
                values = grouped[model][group][:, selected]
                movement = np.full(layers, np.nan)
                for layer in range(1, layers):
                    delta = values[layer] - values[layer - 1]
                    movement[layer] = np.nanmean(delta) if np.isfinite(delta).any() else np.nan
                ax.plot(np.arange(layers), movement, color=color, linestyle=linestyle,
                        linewidth=1.8, label=f"{name} {group}")
        ax.axhline(0, color="black", linewidth=.8)
        ax.axvline(layers - 1.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Prompt {prompt}: green/red mean layer movement")
        ax.set_ylabel("Δ logit")
        ax.grid(alpha=.25)
        ax.legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("hidden-state layer transition (value at l = l - l-1)")
    fig.suptitle("KGW green/red logit-lens movement by prompt")
    fig.tight_layout()
    movement_path = out / "kgw_group_movement_by_prompt.png"
    fig.savefig(movement_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # Histograms of per-position movements for final transitions. Green/red
    # distributions are plotted separately; no averaging over positions.
    transition_layers = list(range(max(1, layers - last_transitions), layers))
    fig, axes = plt.subplots(len(transition_layers), 1, figsize=(11, max(4, 3.4 * len(transition_layers))), squeeze=False)
    for row, layer in enumerate(transition_layers):
        ax = axes[row, 0]
        for model, name in enumerate(model_names):
            color = colors.get(name, ["#2563eb", "#f97316"][model])
            for group, group_color in (("green", "#16a34a"), ("red", "#dc2626")):
                current = grouped[model][group][layer]
                previous = grouped[model][group][layer - 1]
                movement = current - previous
                movement = movement[np.isfinite(movement)]
                if movement.size:
                    label = f"{name} {group} (n={movement.size})"
                    ax.hist(movement, bins=bins, alpha=.38, color=group_color,
                            edgecolor="none", label=label)
            # model colors are encoded in labels; green/red encode KGW class.
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(f"Layer {layer - 1} → {layer}: per-position movement")
        ax.set_xlabel("projected logit movement")
        ax.set_ylabel("count")
        ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Distribution of KGW green/red logit-lens movement")
    fig.tight_layout()
    hist_path = out / "kgw_group_movement_histograms.png"
    fig.savefig(hist_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # Direct final-transition green/red gap distribution, which is often the
    # clearest single view of a KGW-style split.
    final = layers - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, name in enumerate(model_names):
        g = grouped[model]["green"][final] - grouped[model]["green"][final - 1]
        r = grouped[model]["red"][final] - grouped[model]["red"][final - 1]
        gap = g - r
        gap = gap[np.isfinite(gap)]
        if gap.size:
            ax.hist(gap, bins=bins, alpha=.45, label=f"{name} green movement − red movement")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_title(f"Final transition ({final - 1} → {final}) green-minus-red movement")
    ax.set_xlabel("Δgreen − Δred"); ax.set_ylabel("count"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout()
    gap_path = out / "kgw_final_green_red_movement_gap.png"
    fig.savefig(gap_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return movement_path, hist_path, gap_path


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
    model_b_source = metadata.get("model_b", {}).get("source", "k1")
    model_b_name = "k1" if "-k1-" in model_b_source else ("k0" if "-k0-" in model_b_source else "watermarked")
    model_names = ("base", model_b_name)
    rows = rows_for_csv(grouped, prompt_index, model_names)
    out = Path(args.output_dir) if args.output_dir else source / "kgw_group_analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "kgw_group_layer_summary.csv", rows)
    plot_paths = make_plots(out, grouped, prompt_index, model_names, args.last_transitions, args.bins)
    np.savez_compressed(out / "kgw_group_metrics.npz",
                        green=np.stack([grouped[0]["green"], grouped[1]["green"]]),
                        red=np.stack([grouped[0]["red"], grouped[1]["red"]]),
                        prompt_index=prompt_index)
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
