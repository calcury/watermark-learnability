#!/usr/bin/env python
"""Download the Pythia watermark-distilled model and its base model.

Colab usage (run from the repository root)::

    !pip install -q -U transformers huggingface_hub
    !python analysis/fetch_model.py
    !python analysis/diff.py

Both repositories are downloaded directly from Hugging Face into
``pretrained/<model-name>``. ``analysis/diff.py`` then reads only those local
directories, so the analysis step needs no network access. Pythia models are
public, so no token is required; ``--token``/``HF_TOKEN`` is still accepted for
rate limits or private mirrors of the same weights.
"""

import argparse
import os
from pathlib import Path

DEFAULT_WATERMARKED = "cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2"
DEFAULT_BASE = "EleutherAI/pythia-1.4b"

# Formats we never need for PyTorch inference; skipping them saves disk space.
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "*.flax"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watermarked", default=DEFAULT_WATERMARKED, help="Watermark-distilled Pythia repo ID")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base/original Pythia repo ID")
    parser.add_argument("--output-dir", default="pretrained", help="Directory that receives the model folders")
    parser.add_argument("--watermarked-name", default=None, help="Local folder name for the watermarked model")
    parser.add_argument("--base-name", default=None, help="Local folder name for the base model")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face token, or set HF_TOKEN")
    return parser.parse_args()


def default_local_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def download(repo_id: str, target: Path, token: str) -> Path:
    """Download ``repo_id`` into ``target`` (a plain huggingface.co download)."""
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(target),
        "ignore_patterns": IGNORE_PATTERNS,
    }
    if token:
        kwargs["token"] = token
    try:
        path = snapshot_download(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "This huggingface_hub version does not support local_dir downloads. "
            "Run `!pip install -q -U huggingface_hub` and restart the runtime."
        ) from exc
    return Path(path)


def verify(model_dir: Path) -> str:
    """Check the essentials exist and describe the weight files found."""
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"{model_dir} has no config.json; the download was incomplete")
    weights = sorted(
        p.name
        for pattern in ("*.safetensors", "*.bin")
        for p in model_dir.glob(pattern)
    )
    if not weights:
        raise RuntimeError(f"{model_dir} has no weight files (*.safetensors/*.bin)")
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model") if (model_dir / name).is_file()]
    if not tokenizer_files:
        raise RuntimeError(f"{model_dir} has no tokenizer files")
    total_gb = sum((model_dir / name).stat().st_size for name in weights) / 1e9
    return f"{len(weights)} weight file(s), {total_gb:.2f} GB, tokenizer: {', '.join(tokenizer_files)}"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    jobs = [
        ("watermarked", args.watermarked, args.watermarked_name or default_local_name(args.watermarked)),
        ("base", args.base, args.base_name or default_local_name(args.base)),
    ]
    if jobs[0][2] == jobs[1][2]:
        raise ValueError("The watermarked and base models would share a folder name; pass --watermarked-name/--base-name")

    print(f"Downloading into: {output_dir.resolve()}")
    results = {}
    for label, repo_id, folder in jobs:
        target = output_dir / folder
        print(f"\n[{label}] {repo_id} -> {target}")
        path = download(repo_id, target, args.token)
        print(f"Downloaded: {path}")
        print(f"[{label}] {verify(path)}")
        results[label] = path

    print("\nDone. Now run the comparison with:")
    print(
        "  python analysis/diff.py \\\n"
        f"      --watermarked {results['watermarked']} \\\n"
        f"      --base {results['base']}"
    )


if __name__ == "__main__":
    main()
