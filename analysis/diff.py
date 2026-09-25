#!/usr/bin/env python
"""Compare hidden-state distributions of a watermark-distilled Llama and its teacher.

Colab usage (run from the repository root)::

    !pip install -q -U transformers accelerate matplotlib bitsandbytes
    # Restart the Colab runtime after installing/upgrading these packages.
    !python analysis/diff.py

Downloads use ``https://hf-mirror.com`` by default. To use the official
endpoint instead, add ``--hf-endpoint https://huggingface.co``. A mirror can
improve connectivity but cannot bypass a gated model's permission requirement;
for Llama 2, accept the license and set ``HF_TOKEN`` when required.

The default ``--quantization auto`` loads one model at a time in 4-bit on a
Colab GPU. This is important for 12 GB RAM / 15 GB VRAM runtimes. Use
``--quantization none`` only on a machine with enough memory.

The script uses paired prompts for both models and reports per-layer cosine
 distance, normalized L2 distance, and linear CKA. A CSV and diagnostic plots
are written under ``analysis/diff_output`` by default.
"""

import argparse
import csv
import gc
import importlib.metadata
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Keep Transformers lazy-imported: huggingface_hub reads HF_ENDPOINT at import
# time, so setting it after importing Transformers is too late.

DEFAULT_STUDENT = "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
DEFAULT_TEACHER = "meta-llama/Llama-2-7b-hf"
DEFAULT_PROMPTS = [
    "The history of science is a story of people asking questions about the world.",
    "A good education helps people understand their communities and make informed decisions.",
    "In the future, renewable energy may transform how cities are designed.",
    "The recipe calls for fresh vegetables, olive oil, and a little salt.",
    "When building reliable software, testing small components can prevent larger problems.",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=DEFAULT_STUDENT, help="Student/distilled model ID or local path")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER, help="Source Llama model ID or local path")
    parser.add_argument("--prompts", nargs="*", default=None, help="Texts to compare (defaults to built-in prompts)")
    parser.add_argument("--prompt-file", help="UTF-8 text file with one prompt per line")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum tokenized prompt length")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size (keep small for Colab GPUs)")
    parser.add_argument("--max-prompts", type=int, default=5, help="Limit prompts when using --prompt-file")
    parser.add_argument("--output-dir", default="analysis/diff_output", help="Directory for CSV and plots")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--quantization", default="auto", choices=["auto", "4bit", "none"], help="Model loading mode; auto uses 4-bit on CUDA")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"), help="Hugging Face endpoint (default: hf-mirror.com; also sets HF_ENDPOINT)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face access token, or set HF_TOKEN")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code from Hugging Face")
    return parser.parse_args()


def configure_endpoint(endpoint: str):
    """Configure the mirror before importing Transformers/huggingface_hub."""
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ENDPOINT"] = endpoint
    return endpoint


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


def load_model(model_id: str, device: torch.device, trust_remote_code: bool, quantization: str, endpoint: str, token: str):
    """Load one model, preferably quantized, without making a second RAM copy."""
    endpoint = configure_endpoint(endpoint)
    load_kwargs = {"use_fast": True, "trust_remote_code": trust_remote_code}
    if token:
        load_kwargs["token"] = token
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_4bit = quantization == "4bit" or (quantization == "auto" and device.type == "cuda")
    # Set this before Transformers/huggingface_hub perform any downloads.
    os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
    if token:
        kwargs["token"] = token
    if use_4bit:
        if device.type != "cuda":
            raise ValueError("4-bit quantization requires CUDA; use --quantization none on CPU")
        try:
            bnb_version = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "bitsandbytes is not installed in this runtime (or its metadata is stale). "
                "Run `!pip install -q -U bitsandbytes transformers`, then restart the Colab runtime. "
                "Alternatively run with `--quantization none` (uses much more memory)."
            ) from exc
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install a recent transformers and bitsandbytes: !pip install -q -U transformers bitsandbytes") from exc
        print(f"Using 4-bit bitsandbytes quantization (bitsandbytes {bnb_version})")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.float16 if device.type == "cuda" else torch.float32
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not use_4bit:
        model.to(device)
    model.eval()
    return tokenizer, model


def collect_hidden_states(tokenizer, model, prompts: List[str], device, max_length: int, batch_size: int):
    """Return per-layer matrices of masked token activations on CPU."""
    collected: List[List[torch.Tensor]] = [[] for _ in range(model.config.num_hidden_layers + 1)]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        mask = encoded["attention_mask"].bool()
        for layer, state in enumerate(result.hidden_states):
            # Pool each token as a sample; mask padding so lengths do not bias metrics.
            values = state[mask].detach().to(dtype=torch.float32, device="cpu")
            collected[layer].append(values)
        del result, encoded
    return [torch.cat(chunks, dim=0).numpy() for chunks in collected]


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


def compare_layers(student_states, teacher_states) -> List[Dict[str, float]]:
    if len(student_states) != len(teacher_states):
        raise ValueError(f"Models expose different layer counts: {len(student_states)} vs {len(teacher_states)}")
    rows = []
    for layer, (student, teacher) in enumerate(zip(student_states, teacher_states)):
        if student.shape[0] != teacher.shape[0]:
            raise ValueError("Models tokenized prompts to different numbers of tokens; use compatible tokenizers/prompt lengths")
        # Per-token cosine distance, aggregated across all prompt tokens.
        s_norm = np.linalg.norm(student, axis=1).clip(min=1e-12)
        t_norm = np.linalg.norm(teacher, axis=1).clip(min=1e-12)
        cosine_distance = float(np.mean(1.0 - np.sum(student * teacher, axis=1) / (s_norm * t_norm)))
        teacher_rms = float(np.sqrt(np.mean(np.square(teacher))))
        normalized_l2 = float(np.sqrt(np.mean(np.square(student - teacher))) / max(teacher_rms, 1e-12))
        rows.append({
            "layer": layer,
            "cosine_distance": cosine_distance,
            "normalized_l2": normalized_l2,
            "linear_cka": linear_cka(student, teacher),
            "student_rms": float(np.sqrt(np.mean(np.square(student)))),
            "teacher_rms": teacher_rms,
        })
    return rows


def save_results(rows: List[Dict[str, float]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layer_distances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # Lazy import keeps metrics usable in minimal installations; matplotlib is installed in Colab.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title, color in zip(
        axes,
        ("cosine_distance", "normalized_l2", "linear_cka"),
        ("Cosine distance (lower is closer)", "Normalized RMS L2 (lower is closer)", "Linear CKA (higher is more similar)"),
        ("#d95f02", "#1b9e77", "#7570b3"),
    ):
        ax.plot(layers, [row[key] for row in rows], marker="o", linewidth=1.8, color=color)
        ax.set_xlabel("Hidden-state layer (0 = embeddings)")
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Student vs source model representation differences")
    fig.tight_layout()
    plot_path = output_dir / "layer_distances.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return csv_path, plot_path


def main():
    args = parse_args()
    # Must happen before the lazy Transformers import in load_model.
    args.hf_endpoint = configure_endpoint(args.hf_endpoint)
    prompts = get_prompts(args)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"Device: {device}; prompts: {len(prompts)}; max length: {args.max_length}")
    print(f"Loading student: {args.student}")
    print(f"Hugging Face endpoint: {args.hf_endpoint}")
    student_tokenizer, student_model = load_model(args.student, device, args.trust_remote_code, args.quantization, args.hf_endpoint, args.token)
    student_states = collect_hidden_states(student_tokenizer, student_model, prompts, device, args.max_length, args.batch_size)
    del student_model, student_tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Loading teacher: {args.teacher}")
    teacher_tokenizer, teacher_model = load_model(args.teacher, device, args.trust_remote_code, args.quantization, args.hf_endpoint, args.token)
    teacher_states = collect_hidden_states(teacher_tokenizer, teacher_model, prompts, device, args.max_length, args.batch_size)
    rows = compare_layers(student_states, teacher_states)
    print("\nPer-layer representation differences (token-weighted over prompts):")
    print(f"{'Layer':>5} {'Cosine dist':>13} {'Norm. L2':>12} {'Linear CKA':>12} {'Student RMS':>13} {'Teacher RMS':>13}")
    for row in rows:
        print(f"{row['layer']:5d} {row['cosine_distance']:13.6f} {row['normalized_l2']:12.6f} {row['linear_cka']:12.6f} {row['student_rms']:13.6f} {row['teacher_rms']:13.6f}")
    csv_path, plot_path = save_results(rows, Path(args.output_dir))
    print(f"\nSaved metrics: {csv_path}")
    print(f"Saved plot:    {plot_path}")


if __name__ == "__main__":
    main()
