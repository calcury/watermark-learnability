#!/usr/bin/env python
"""Summarize and visualize ``delta_logits.npz`` from compare_llama_logits.py.

Colab example::

    !python analysis/analyze_logit_diff.py \
        --input-dir analysis/llama_logit_diff_output

The script loads the full matrix ``delta = watermarked_logits - base_logits``
and reproduces the KGW green/red mask for each final prompt context. It writes
per-prompt summary, token-level extrema, distributions, a prompt-by-token
heatmap, and KGW tests: centered mask/delta cosine and the green-minus-red
mean shift compared with the configured KGW bias.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="analysis/llama_logit_diff_output")
    p.add_argument("--top-k", type=int, default=20, help="Rows per direction and prompt in the token table")
    p.add_argument("--heatmap-tokens", type=int, default=100, help="Most shifted tokens shown in heatmap")
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--prompt-index", type=int, default=None,
                   help="Analyze only one prompt, using its zero-based index (for example --prompt-index 2)")
    p.add_argument("--normalization", choices=["none", "center", "zscore"], default="none",
                   help="Optional per-prompt transform before plotting (default: keep raw delta logits)")
    p.add_argument("--gamma", type=float, default=0.25, help="KGW green-list fraction")
    p.add_argument("--bias", type=float, default=2.0, help="KGW bias delta used for comparison")
    p.add_argument("--seeding-scheme", default="simple_1", help="KGW seeding scheme, usually simple_1")
    p.add_argument("--hf-token", default=None, help="Optional Hugging Face token for loading a gated tokenizer")
    p.add_argument("--max-length", type=int, default=256,
                   help="Must match --max-length used when producing delta_logits.npz")
    return p.parse_args()


def load_data(input_dir):
    input_dir = Path(input_dir)
    data = np.load(input_dir / "delta_logits.npz")
    delta = np.asarray(data["delta"], dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError(f"Expected a 2-D delta matrix, got shape {delta.shape}")
    metadata_path = input_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    prompts = metadata.get("prompts", [f"prompt_{i}" for i in range(delta.shape[0])])
    if len(prompts) != delta.shape[0]:
        prompts = [f"prompt_{i}" for i in range(delta.shape[0])]
    return input_dir, delta, prompts, metadata


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def token_text(tokenizer, token_id):
    if tokenizer is None:
        return ""
    return tokenizer.decode([int(token_id)]).replace("\n", "\\n")


def build_tables(output_dir, delta, prompts, top_k, tokenizer):
    rows = []
    summary = []
    k = min(max(1, top_k), delta.shape[1])
    for i, row in enumerate(delta):
        positive = np.argpartition(row, -k)[-k:][::-1]
        negative = np.argpartition(row, k - 1)[:k]
        for direction, ids in (("positive", positive), ("negative", negative)):
            for rank, token_id in enumerate(ids, 1):
                rows.append({"prompt_index": i, "direction": direction, "rank": rank,
                             "token_id": int(token_id), "token": token_text(tokenizer, token_id),
                             "delta_logit": float(row[token_id]), "prompt": prompts[i]})
        summary.append({"prompt_index": i, "prompt": prompts[i],
                        "mean_delta": float(row.mean()),
                        "median_delta": float(np.median(row)),
                        "rms_delta": float(np.sqrt(np.mean(row ** 2))),
                        "std_delta": float(row.std()),
                        "p01": float(np.quantile(row, .01)),
                        "p99": float(np.quantile(row, .99)),
                        "positive_tokens": int(np.sum(row > 0)),
                        "negative_tokens": int(np.sum(row < 0)),
                        "max_positive": float(row.max()),
                        "min_negative": float(row.min())})
    write_csv(output_dir / "token_shift_table.csv", rows)
    write_csv(output_dir / "prompt_summary.csv", summary)
    return summary, rows


def kgw_masks(input_ids, vocab_size, gamma, seeding_scheme, tokenizer=None):
    """Reproduce KGW green masks for the final next-token position of each prompt."""
    import torch
    from watermarks.kgw.watermark_processor import WatermarkBase

    if not 0 < gamma < 1:
        raise ValueError("--gamma must be between 0 and 1")
    wm = WatermarkBase(vocab=list(range(vocab_size)), gamma=gamma,
                       seeding_scheme=seeding_scheme, device="cpu")
    if tokenizer is not None and seeding_scheme == "simple_1":
        # Match KGWWatermark's special-token exclusion where possible.
        special_ids = {x for x in (tokenizer.eos_token_id, tokenizer.bos_token_id,
                                    tokenizer.pad_token_id, tokenizer.unk_token_id)
                       if x is not None and 0 <= x < vocab_size}
    else:
        special_ids = set()
    masks = []
    for ids in input_ids:
        context = ids.detach().cpu().long()
        if context.numel() < wm.context_width:
            raise ValueError(f"Prompt has fewer than {wm.context_width} tokens required by {seeding_scheme}")
        green = wm._get_greenlist_ids(context)
        mask = np.zeros(vocab_size, dtype=bool)
        mask[green.numpy()] = True
        mask[list(special_ids)] = False
        masks.append(mask)
    return np.asarray(masks, dtype=bool)


def mask_statistics(delta, masks, prompts, bias):
    rows = []
    for i, (row, green) in enumerate(zip(delta, masks)):
        red = ~green
        centered_delta = row - row.mean()
        centered_mask = green.astype(np.float64) - green.mean()
        denominator = np.linalg.norm(centered_delta) * np.linalg.norm(centered_mask)
        cosine = float(np.dot(centered_delta, centered_mask) / denominator) if denominator else float("nan")
        green_mean = float(row[green].mean()) if green.any() else float("nan")
        red_mean = float(row[red].mean()) if red.any() else float("nan")
        rows.append({"prompt_index": i, "prompt": prompts[i], "green_tokens": int(green.sum()),
                     "red_tokens": int(red.sum()), "green_fraction": float(green.mean()),
                     "green_mean_delta": green_mean, "red_mean_delta": red_mean,
                     "green_minus_red": green_mean - red_mean,
                     "kgw_bias": float(bias), "gap_minus_bias": green_mean - red_mean - bias,
                     "centered_mask_delta_cosine": cosine,
                     "centered_delta_mean": float(centered_delta.mean()),
                     "centered_delta_std": float(centered_delta.std())})
    return rows


def normalize_delta(delta, method):
    """Normalize each prompt independently across its vocabulary dimension."""
    centered = delta - delta.mean(axis=1, keepdims=True)
    if method == "none":
        return delta
    if method == "center":
        return centered
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def make_plots(output_dir, delta, prompts, heatmap_tokens, bins, top_k, normalization):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_prompts = delta.shape[0]
    labels = [f"p{i}" for i in range(n_prompts)]

    # Distribution of all vocabulary shifts for each prompt. We remove the
    # per-prompt global offset, then optionally divide by its standard deviation.
    normalized = normalize_delta(delta, normalization)
    np.save(output_dir / f"delta_{normalization}.npy", normalized)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, row in enumerate(normalized):
        axes[0].hist(row, bins=bins, alpha=.45, density=True, label=labels[i])
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(title=f"{normalization.title()} vocabulary logit shifts", xlabel=f"{normalization} delta logit", ylabel="density")
    axes[0].legend()
    axes[1].boxplot(normalized.T, labels=labels, showfliers=False)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title=f"{normalization.title()} shift by prompt", xlabel="prompt", ylabel=f"{normalization} delta logit")
    fig.tight_layout(); fig.savefig(output_dir / "shift_distributions.png", dpi=180); plt.close(fig)

    # Show the globally largest absolute shifts, preserving prompt rows.
    count = min(heatmap_tokens, delta.shape[1])
    ids = np.argsort(np.max(np.abs(delta), axis=0))[-count:][::-1]
    fig, ax = plt.subplots(figsize=(max(10, count * .12), max(4, n_prompts * .75)))
    image = ax.imshow(delta[:, ids], aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(delta[:, ids])), vmax=np.max(np.abs(delta[:, ids])))
    ax.set(yticks=np.arange(n_prompts), yticklabels=labels, xlabel="tokens sorted by max |delta|", title="Watermarked minus clean logits")
    ax.set_xticks([])
    fig.colorbar(image, ax=ax, label="delta logit")
    fig.tight_layout(); fig.savefig(output_dir / "delta_heatmap.png", dpi=180); plt.close(fig)

    # Per-prompt RMS and extrema.
    rms = np.sqrt(np.mean(delta ** 2, axis=1))
    max_pos = delta.max(axis=1)
    min_neg = delta.min(axis=1)
    x = np.arange(n_prompts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rms, "o-", label="RMS")
    ax.plot(x, max_pos, "^--", label="max positive")
    ax.plot(x, min_neg, "v--", label="min negative")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xticks=x, xticklabels=labels, title="Magnitude of logit changes", ylabel="logit shift")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output_dir / "shift_magnitude.png", dpi=180); plt.close(fig)

    # Overlap of top positive tokens between prompts (a simple green-list-like check).
    top_ids = [set(np.argpartition(row, -top_k)[-top_k:]) for row in delta]
    overlap = np.zeros((n_prompts, n_prompts), dtype=float)
    for i in range(n_prompts):
        for j in range(n_prompts):
            overlap[i, j] = len(top_ids[i] & top_ids[j]) / max(1, top_k)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(overlap, vmin=0, vmax=1, cmap="viridis")
    ax.set(xticks=x, xticklabels=labels, yticks=x, yticklabels=labels,
           title=f"Overlap of top-{top_k} positive-shift tokens", xlabel="prompt", ylabel="prompt")
    fig.colorbar(image, ax=ax, label="fraction shared")
    fig.tight_layout(); fig.savefig(output_dir / "positive_token_overlap.png", dpi=180); plt.close(fig)


def main():
    ns = parse_args()
    output_dir, delta, prompts, metadata = load_data(ns.input_dir)
    original_indices = list(range(delta.shape[0]))
    if ns.prompt_index is not None:
        if not 0 <= ns.prompt_index < delta.shape[0]:
            raise ValueError(f"--prompt-index must be between 0 and {delta.shape[0] - 1}")
        delta = delta[ns.prompt_index:ns.prompt_index + 1]
        prompts = [prompts[ns.prompt_index]]
        original_indices = [ns.prompt_index]
        print(f"Filtering to p{ns.prompt_index}: {prompts[0]}")
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        source = metadata.get("base")
        if source:
            tokenizer = AutoTokenizer.from_pretrained(
                source, token=ns.hf_token, use_fast=True,
                local_files_only=Path(source).is_dir())
    except Exception as exc:
        print(f"Tokenizer unavailable; token IDs will still be reported ({exc})")

    if tokenizer is None:
        raise RuntimeError("A tokenizer is required for KGW mask analysis; pass --hf-token or use a local base model")

    summary, _ = build_tables(output_dir, delta, prompts, ns.top_k, tokenizer)
    # Re-tokenize the same prompts with the base tokenizer and reproduce the KGW
    # green list for the final context used to obtain each delta vector.
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=ns.max_length)
    final_ids = []
    for row_ids, row_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
        final_ids.append(row_ids[row_mask.bool()])
    masks = kgw_masks(final_ids, delta.shape[1], ns.gamma, ns.seeding_scheme, tokenizer)
    mask_rows = mask_statistics(delta, masks, prompts, ns.bias)
    write_csv(output_dir / "kgw_mask_analysis.csv", mask_rows)
    print("\nKGW mask analysis:")
    print(f"{'prompt':>8} {'green_mean':>12} {'red_mean':>12} {'gap':>10} {'bias':>10} {'cosine':>10}")
    for row in mask_rows:
        print(f"p{row['prompt_index']:>6} {row['green_mean_delta']:12.5f} {row['red_mean_delta']:12.5f} "
              f"{row['green_minus_red']:10.5f} {row['kgw_bias']:10.5f} {row['centered_mask_delta_cosine']:10.5f}")
    make_plots(output_dir, delta, prompts, ns.heatmap_tokens, ns.bins, ns.top_k, ns.normalization)
    print(f"Loaded delta matrix: {delta.shape}")
    print("\nPer-prompt summary:")
    print(f"{'prompt':>8} {'RMS':>10} {'mean':>10} {'p01':>10} {'p99':>10} {'max+':>10} {'min-':>10}")
    for row in summary:
        print(f"p{row['prompt_index']:>6} {row['rms_delta']:10.5f} {row['mean_delta']:10.5f} "
              f"{row['p01']:10.5f} {row['p99']:10.5f} {row['max_positive']:10.5f} {row['min_negative']:10.5f}")
    print(f"\nWrote tables and plots under {output_dir}")


if __name__ == "__main__":
    main()
