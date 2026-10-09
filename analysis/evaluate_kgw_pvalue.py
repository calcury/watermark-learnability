#!/usr/bin/env python3
"""Generate text from a model and report KGW p-values for generated tokens only."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scipy.stats import binom
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from watermarks.kgw.watermark_processor import WatermarkDetector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model ID or local model folder")
    p.add_argument("--prompts", required=True, help="UTF-8 file with one prompt per line")
    p.add_argument("--output", required=True, help="JSONL output for generations and p-values")
    p.add_argument("--k", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--delta", type=int, choices=(1, 2), default=1,
                   help="Select KGW config matching generation; delta does not change detector formula")
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--seeding-scheme", choices=("simple_0", "simple_1", "simple_2"), default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    scheme = args.seeding_scheme or f"simple_{args.k}"
    prompts = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Prompt file contains no non-empty lines")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit or args.load_in_8bit:
        if device != "cuda":
            raise RuntimeError("--load-in-4bit/8bit requires --device cuda")
        load_kwargs.update({
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        })
    else:
        load_kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not (args.load_in_4bit or args.load_in_8bit):
        model.to(device)
    model.eval()
    detector = WatermarkDetector(
        vocab=list(range(len(tokenizer))), gamma=args.gamma,
        seeding_scheme=scheme, tokenizer=tokenizer, device=torch.device("cpu"),
        normalizers=[],
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output_path.open("w", encoding="utf-8") as out, torch.inference_mode():
        for index, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5), top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = encoded["input_ids"].shape[1]
            generated_ids = generated[0, prompt_len:].detach().cpu().tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Score only generated tokens; prompt IDs are used only as KGW seed context.
            context = encoded["input_ids"][0].detach().cpu().tolist()
            green_count = 0
            scored_count = 0
            for token_id in generated_ids:
                if len(context) < detector.context_width:
                    context.append(token_id)
                    continue
                if token_id in tokenizer.all_special_ids:
                    # Decoded text omits special IDs, so they are not observable
                    # to text-based detection; retain them only in subsequent context.
                    context.append(token_id)
                    continue
                green_ids = detector._get_greenlist_ids(torch.tensor(context, dtype=torch.long))
                green_count += int(token_id in green_ids.tolist())
                scored_count += 1
                context.append(token_id)
            if scored_count:
                p_value = float(binom.sf(green_count - 1, scored_count, args.gamma))
                z_score = float((green_count - args.gamma * scored_count) /
                                (scored_count * args.gamma * (1 - args.gamma)) ** 0.5)
            else:
                p_value, z_score = 1.0, 0.0
            record = {
                "prompt_index": index, "prompt": prompt, "generated_text": generated_text,
                "k": args.k, "delta": args.delta, "gamma": args.gamma,
                "seeding_scheme": scheme, "generated_tokens_scored": scored_count,
                "green_tokens": green_count, "green_fraction": (green_count / scored_count if scored_count else None),
                "z_score": z_score, "p_value": p_value,
            }
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}] tokens={scored_count} green={green_count} z={z_score:.4f} p={p_value:.6g}")
    if records:
        pvals = [r["p_value"] for r in records]
        summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
        summary_path.write_text(json.dumps({
            "model": args.model, "k": args.k, "delta": args.delta, "gamma": args.gamma,
            "seeding_scheme": scheme, "prompt_count": len(records),
            "median_p_value": (sorted(pvals)[(len(pvals) - 1) // 2] + sorted(pvals)[len(pvals) // 2]) / 2,
            "mean_p_value": sum(pvals) / len(pvals), "records": records,
            "note": "P-values score generated continuation tokens only; the prompt is used as seed context."
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
