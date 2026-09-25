#!/usr/bin/env python
"""Compare layerwise representations of a watermark-distilled model and its base model.

Colab usage (run from the repository root)::

    !pip install -q -U transformers matplotlib
    !python analysis/fetch_model.py   # downloads to pretrained/
    !python analysis/diff.py

``diff.py`` never touches the network: it loads the two models from the local
``pretrained/`` directories created by ``fetch_model.py``. Both Pythia models
are small (about 2.8 GB in fp16), and they are still loaded one at a time so the
run stays inside a 12 GB RAM / 15 GB VRAM Colab runtime.

For every hidden-state layer the script reports the distribution of per-token
cosine distance, normalized L2 distance, and linear CKA, prints a table, and
writes ``layer_distances.csv`` plus two figures under ``analysis/diff_output``.
"""

import argparse
import csv
import gc
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

DEFAULT_WATERMARKED = "pretrained/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2"
DEFAULT_BASE = "pretrained/pythia-1.4b"
DEFAULT_PROMPTS = [
    "The history of science is a story of people asking questions about the world.",
    "A good education helps people understand their communities and make informed decisions.",
    "In the future, renewable energy may transform how cities are designed.",
    "The recipe calls for fresh vegetables, olive oil, and a little salt.",
    "When building reliable software, testing small components can prevent larger problems.",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watermarked", default=DEFAULT_WATERMARKED, help="Local directory of the watermark-distilled model")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Local directory of the base/original model")
    parser.add_argument("--prompts", nargs="*", default=None, help="Texts to compare (defaults to built-in prompts)")
    parser.add_argument("--prompt-file", help="UTF-8 text file with one prompt per line")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum tokenized prompt length")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size (keep small for Colab GPUs)")
    parser.add_argument("--max-prompts", type=int, default=5, help="Limit prompts when using --prompt-file")
    parser.add_argument("--output-dir", default="analysis/diff_output", help="Directory for CSV and plots")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code from the local directory")
    return parser.parse_args()


def get_prompts(args) -> List[str]:
    if args.prompts:
        prompts = args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        prompts = prompts[: args.max_prompts]
    else:
        prompts = DEFAULT_PROMPTS[: args.max_prompts]
    if not prompts:
        raise ValueError("No prompts to analyze")
    return prompts


def check_local_dir(path: str, flag: str) -> Path:
    model_dir = Path(path)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"{flag} directory not found: {model_dir}\n"
            "Run `python analysis/fetch_model.py` first to download the models into pretrained/."
        )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"{model_dir} has no config.json; re-run `python analysis/fetch_model.py`")
    return model_dir


def load_model(model_dir: Path, device: torch.device, trust_remote_code: bool):
    """Load one model from disk in fp16 on CUDA / fp32 on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def tokenize_prompts(tokenizer, prompts: Sequence[str], max_length: int, batch_size: int):
    """Tokenize once so both models see byte-identical inputs."""
    batches = []
    for start in range(0, len(prompts), batch_size):
        batches.append(
            tokenizer(
                list(prompts[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        )
    return batches


def check_same_tokenization(base_batches, watermarked_batches):
    for index, (base, watermarked) in enumerate(zip(base_batches, watermarked_batches)):
        for key in ("input_ids", "attention_mask"):
            if not torch.equal(base[key], watermarked[key]):
                raise RuntimeError(
                    f"The two models tokenize prompt batch {index} differently ({key}); "
                    "a token-level comparison would not be valid."
                )


def collect_hidden_states(model, batches, device: torch.device) -> List[np.ndarray]:
    """Return per-layer matrices of masked token activations on CPU."""
    collected: List[List[torch.Tensor]] = [[] for _ in range(model.config.num_hidden_layers + 1)]
    for encoded in batches:
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        mask = inputs["attention_mask"].bool()
        for layer, state in enumerate(result.hidden_states):
            # Every non-padding token is one sample; padding is masked out.
            collected[layer].append(state[mask].detach().to(dtype=torch.float32, device="cpu"))
        del result, inputs
    return [torch.cat(chunks, dim=0).numpy() for chunks in collected]


def per_token_cosine_distance(student: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    s_norm = np.linalg.norm(student, axis=1).clip(min=1e-12)
    t_norm = np.linalg.norm(teacher, axis=1).clip(min=1e-12)
    return (1.0 - np.sum(student * teacher, axis=1) / (s_norm * t_norm)).astype(np.float64)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between paired activation rows; invariant to isotropic scaling."""
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denom = np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro")
    return float(np.sum(cross * cross) / denom) if denom > 0 else float("nan")


def compare_layers(
    student_states: Sequence[np.ndarray], teacher_states: Sequence[np.ndarray]
) -> Tuple[List[Dict[str, float]], List[np.ndarray]]:
    """Return per-layer summary rows and the per-token cosine distance arrays."""
    if len(student_states) != len(teacher_states):
        raise ValueError(f"Models expose different layer counts: {len(student_states)} vs {len(teacher_states)}")
    rows: List[Dict[str, float]] = []
    distributions: List[np.ndarray] = []
    for layer, (student, teacher) in enumerate(zip(student_states, teacher_states)):
        if student.shape != teacher.shape:
            raise ValueError(f"Layer {layer} shapes differ: {student.shape} vs {teacher.shape}")
        student = student.astype(np.float64, copy=False)
        teacher = teacher.astype(np.float64, copy=False)
        distances = per_token_cosine_distance(student, teacher)
        teacher_rms = float(np.sqrt(np.mean(np.square(teacher))))
        rows.append({
            "layer": layer,
            "cosine_distance_mean": float(distances.mean()),
            "cosine_distance_std": float(distances.std()),
            "cosine_distance_p05": float(np.quantile(distances, 0.05)),
            "cosine_distance_p50": float(np.quantile(distances, 0.50)),
            "cosine_distance_p95": float(np.quantile(distances, 0.95)),
            "normalized_l2": float(np.sqrt(np.mean(np.square(student - teacher))) / max(teacher_rms, 1e-12)),
            "linear_cka": linear_cka(student, teacher),
            "student_rms": float(np.sqrt(np.mean(np.square(student)))),
            "teacher_rms": teacher_rms,
            "tokens": int(student.shape[0]),
        })
        distributions.append(distances)
    return rows, distributions


def save_results(rows: List[Dict[str, float]], distributions: List[np.ndarray], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layer_distances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in rows]

    # Figure 1: how each summary metric evolves with depth.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title, color in zip(
        axes,
        ("cosine_distance_mean", "normalized_l2", "linear_cka"),
        ("Cosine distance (lower is closer)", "Normalized RMS L2 (lower is closer)", "Linear CKA (higher is more similar)"),
        ("#d95f02", "#1b9e77", "#7570b3"),
    ):
        ax.plot(layers, [row[key] for row in rows], marker="o", linewidth=1.8, color=color)
        ax.set_xlabel("Hidden-state layer (0 = embeddings)")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Watermark-distilled Pythia vs base Pythia: per-layer differences")
    fig.tight_layout()
    metrics_path = output_dir / "layer_metrics.png"
    fig.savefig(metrics_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: the per-token cosine-distance distribution of every layer.
    fig2, ax2 = plt.subplots(figsize=(max(8, 0.45 * len(layers)), 6))
    ax2.boxplot(distributions, positions=layers, widths=0.6, showfliers=False, patch_artist=True,
                boxprops={"facecolor": "#a6cee3", "edgecolor": "#1f78b4"},
                medianprops={"color": "#d95f02", "linewidth": 1.6},
                whiskerprops={"color": "#1f78b4"}, capprops={"color": "#1f78b4"})
    ax2.set_xlabel("Hidden-state layer (0 = embeddings)")
    ax2.set_ylabel("Per-token cosine distance to base model")
    ax2.set_title("Layerwise distribution of representation distances (box = IQR, whiskers = 5-95%)")
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    distribution_path = output_dir / "layer_distance_distribution.png"
    fig2.savefig(distribution_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)

    return csv_path, metrics_path, distribution_path


def main():
    args = parse_args()
    watermarked_dir = check_local_dir(args.watermarked, "--watermarked")
    base_dir = check_local_dir(args.base, "--base")
    prompts = get_prompts(args)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"Device: {device}; prompts: {len(prompts)}; max length: {args.max_length}")
    print(f"Watermarked model: {watermarked_dir}")
    print(f"Base model:        {base_dir}")

    # Tokenize with both tokenizers to prove the comparison is well defined.
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True, local_files_only=True)
    wm_tokenizer = AutoTokenizer.from_pretrained(watermarked_dir, use_fast=True, local_files_only=True)
    for tokenizer in (base_tokenizer, wm_tokenizer):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    batches = tokenize_prompts(base_tokenizer, prompts, args.max_length, args.batch_size)
    check_same_tokenization(tokenize_prompts(wm_tokenizer, prompts, args.max_length, args.batch_size), batches)
    tokens = sum(int(batch["attention_mask"].sum()) for batch in batches)
    print(f"Tokenized {len(prompts)} prompts; {tokens} non-padding tokens compared per layer")
    del base_tokenizer, wm_tokenizer

    print("Loading watermarked model...")
    _, watermarked_model = load_model(watermarked_dir, device, args.trust_remote_code)
    watermarked_states = collect_hidden_states(watermarked_model, batches, device)
    num_layers = len(watermarked_states)
    del watermarked_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading base model...")
    _, base_model = load_model(base_dir, device, args.trust_remote_code)
    base_states = collect_hidden_states(base_model, batches, device)
    del base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows, distributions = compare_layers(watermarked_states, base_states)
    print(f"\nLayerwise differences over {num_layers} layers (watermarked vs base, token-weighted):")
    print(f"{'Layer':>5} {'CosDist mean':>13} {'std':>8} {'p05':>8} {'p50':>8} {'p95':>8} {'NormL2':>9} {'CKA':>8}")
    for row in rows:
        print(
            f"{row['layer']:5d} {row['cosine_distance_mean']:13.6f} {row['cosine_distance_std']:8.4f} "
            f"{row['cosine_distance_p05']:8.4f} {row['cosine_distance_p50']:8.4f} {row['cosine_distance_p95']:8.4f} "
            f"{row['normalized_l2']:9.6f} {row['linear_cka']:8.6f}"
        )
    csv_path, metrics_path, distribution_path = save_results(rows, distributions, Path(args.output_dir))
    print(f"\nSaved metrics:               {csv_path}")
    print(f"Saved per-layer metric plot: {metrics_path}")
    print(f"Saved distribution plot:     {distribution_path}")


if __name__ == "__main__":
    main()
