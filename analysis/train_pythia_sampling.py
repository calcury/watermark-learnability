#!/usr/bin/env python3
"""Train Pythia-1.4B by sampling-based KGW distillation.

Examples:
  python analysis/train_pythia_sampling.py --k 0 --delta 1 --mode baseline
  python analysis/train_pythia_sampling.py --k 0 --delta 1 --mode align --alignment-loss-weight 0.1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = "EleutherAI/pythia-1.4b"
DATASET_TEMPLATE = "cygu/sampling-distill-train-data-kgw-k{k}-gamma0.25-delta{delta}"
CONFIG_TEMPLATE = "experiments/watermark-configs/kgw-k{k}-gamma0.25-delta{delta}-config.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--delta", type=int, choices=(1, 2), default=1)
    p.add_argument("--mode", choices=("baseline", "align"), required=True,
                   help="baseline uses the original sampling-distillation CE objective; align adds cosine loss")
    p.add_argument("--alignment-loss-weight", type=float, default=0.0)
    p.add_argument("--reference-model", default=None,
                   help="Clean reference model (defaults to the selected Pythia base checkpoint)")
    p.add_argument("--model", default="pretrained/pythia-1.4b", help="Pythia student initialization (default: fetched local base)")
    p.add_argument("--fetch-base", action=argparse.BooleanOptionalAction, default=True,
                   help="Use analysis/fetch_pythia.py to fetch the default local base if missing")
    p.add_argument("--dataset", default=None, help="Override the sampling-distillation Hugging Face dataset ID")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--optim", default="adafactor",
                   help="Optimizer; adafactor is the low-memory default for 16 GB-class GPUs")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="Trade compute for lower activation memory")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--alignment-positions-per-sequence", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "align" and args.alignment_loss_weight <= 0:
        args.alignment_loss_weight = 0.1

    if args.mode == "baseline":
        args.alignment_loss_weight = 0.0
    if args.fetch_base and args.model == "pretrained/pythia-1.4b" and not (ROOT / args.model).exists():
        subprocess.run([sys.executable, str(ROOT / "analysis" / "fetch_pythia.py"), "base"],
                       cwd=ROOT, check=True, env=os.environ.copy())
    config = ROOT / CONFIG_TEMPLATE.format(k=args.k, delta=args.delta)
    if not config.is_file():
        # Some k/delta combinations have no checked-in config; derive the canonical KGW settings.
        config = ROOT / "analysis" / "generated_watermark_configs" / f"kgw-k{args.k}-gamma0.25-delta{args.delta}-config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({
            "type": "kgw", "k": 1, "gamma": 0.25, "delta": float(args.delta),
            "seeding_scheme": f"simple_{args.k}", "kgw_device": "cuda"
        }, indent=2) + "\n", encoding="utf-8")
    if args.k == 2 and args.delta == 1 and args.dataset is None:
        raise ValueError("No verified default k=2, delta=1 sampling dataset; provide --dataset explicitly")
    dataset = args.dataset or DATASET_TEMPLATE.format(k=args.k, delta=args.delta)
    mode_suffix = "align" if args.mode == "align" else "baseline"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "analysis" / "trained" / f"pythia-1.4b-k{args.k}-delta{args.delta}-{mode_suffix}"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    command = [sys.executable, str(ROOT / "train_sampling_distill.py"),
        "--model_name_or_path", args.model, "--dataset_name", dataset,
        "--watermark_config_file", str(config), "--do_train",
        "--output_dir", str(output_dir), "--learning_rate", str(args.learning_rate),
        "--lr_scheduler_type", "cosine", "--warmup_steps", str(args.warmup_steps),
        "--block_size", str(args.block_size), "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--optim", args.optim,
        "--num_train_epochs", str(args.epochs), "--group_texts", "True",
        "--seed", str(args.seed), "--alignment_loss_weight", str(args.alignment_loss_weight),
        "--alignment_reference_device", args.reference_device,
        "--alignment_positions_per_sequence", str(args.alignment_positions_per_sequence)]
    if args.mode == "align":
        command.extend(["--alignment_reference_model", args.reference_model or args.model])
    if args.max_train_samples is not None:
        command.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    if args.gradient_checkpointing:
        command.extend(["--gradient_checkpointing", "True"])
    if args.bf16:
        command.extend(["--bf16", "True"])
    if args.fp16:
        command.extend(["--fp16", "True"])
    if args.overwrite:
        command.append("--overwrite_output_dir")
    print(f"mode={args.mode}; k={args.k}; delta={args.delta}; dataset={dataset}; output={output_dir}")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
