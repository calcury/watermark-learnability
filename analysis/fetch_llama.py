#!/usr/bin/env python
"""Download Llama-2-7B and its three KGW watermark-distilled variants.

Colab usage (run from the repository root)::

    !pip install -q -U transformers huggingface_hub
    !python analysis/fetch_llama.py base --token YOUR_HF_TOKEN
    !python analysis/fetch_llama.py k0 --token YOUR_HF_TOKEN

Pass exactly one model value (``base``, ``k0``, ``k1``, or ``k2``) per
invocation so that only one model is downloaded at a time. The base Llama repository is gated and requires a Hugging Face token with
access. Models are downloaded into ``pretrained/<model-name>``.
"""

import argparse
import os
from pathlib import Path

DEFAULT_BASE = "meta-llama/Llama-2-7b-hf"
DEFAULT_K0 = "cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2"
DEFAULT_K1 = "cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2"
DEFAULT_K2 = "cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2"

# Formats we never need for PyTorch inference; skipping them saves disk space.
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.tflite", "*.flax"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("value", choices=("base", "k0", "k1", "k2"), help="Model to download")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base/original Llama repo ID")
    parser.add_argument("--k0", default=DEFAULT_K0, help="k=0 watermark-distilled Llama repo ID")
    parser.add_argument("--k1", default=DEFAULT_K1, help="k=1 watermark-distilled Llama repo ID")
    parser.add_argument("--k2", default=DEFAULT_K2, help="k=2 watermark-distilled Llama repo ID")
    parser.add_argument("--output-dir", default="pretrained", help="Directory that receives the model folders")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token, or set HF_TOKEN")
    return parser.parse_args()


def default_local_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def download(repo_id: str, target: Path, token: str) -> Path:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": repo_id, "local_dir": str(target), "ignore_patterns": IGNORE_PATTERNS}
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
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"{model_dir} has no config.json; the download was incomplete")
    # Verify every checkpoint shard when an index is present.
    index_files = [model_dir / "model.safetensors.index.json",
                   model_dir / "pytorch_model.bin.index.json"]
    indexes = [path for path in index_files if path.is_file()]
    if indexes:
        index = indexes[0]
        try:
            import json
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid checkpoint index {index}: {exc}") from exc
        weights = sorted(set(weight_map.values()))
        if not weights:
            raise RuntimeError(f"{index} has no weight_map entries")
        missing = [name for name in weights if not (model_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{model_dir} is missing checkpoint shard(s): {', '.join(missing)}")
    else:
        weights = sorted(p.name for pattern in ("*.safetensors", "*.bin") for p in model_dir.glob(pattern))
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
    repositories = {"base": args.base, "k0": args.k0, "k1": args.k1, "k2": args.k2}
    label, repo_id = args.value, repositories[args.value]
    target = output_dir / default_local_name(repo_id)

    print(f"Downloading into: {output_dir.resolve()}")
    print(f"\n[{label}] {repo_id} -> {target}")
    path = download(repo_id, target, args.token)
    print(f"Downloaded: {path}")
    print(f"[{label}] {verify(path)}")


if __name__ == "__main__":
    main()
