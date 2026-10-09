#!/usr/bin/env python3
"""Generate continuations and save KGW-labelled hidden states in checkpoints.

The extractor writes small chunk files and a lightweight manifest instead of
keeping all Llama-2 hidden states in RAM. This is important for 7B models.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from watermarks.kgw.watermark_processor import WatermarkDetector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--texts", required=True)
    p.add_argument("--output", required=True, help="Manifest .pt path")
    p.add_argument("--k", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--delta", type=int, choices=(1, 2), default=2)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--seeding-scheme", default=None)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--min-context", type=int, default=None)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=5,
                   help="Prompts per checkpoint chunk; lower this if Colab RAM is killed")
    p.add_argument("--generations-output", default=None)
    p.add_argument("--input-generations", default=None,
                   help="Reuse existing generations JSONL and skip model.generate")
    return p.parse_args()


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"trust_remote_code": args.trust_remote_code}
    quantized = args.load_in_4bit or args.load_in_8bit
    if quantized:
        if device != "cuda":
            raise RuntimeError("4/8-bit loading requires --device cuda")
        kwargs.update({
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
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not quantized:
        model.to(device)
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    texts = [x.strip() for x in Path(args.texts).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not texts and not args.input_generations:
        raise ValueError("--texts contains no non-empty lines")
    model, tokenizer = load_model(args, device)
    scheme = args.seeding_scheme or f"simple_{args.k}"
    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()), gamma=args.gamma,
        seeding_scheme=scheme, tokenizer=tokenizer,
        device=torch.device("cpu"), normalizers=[])
    min_context = args.min_context or detector.context_width

    if args.input_generations:
        records = [json.loads(x) for x in Path(args.input_generations).read_text(encoding="utf-8").splitlines() if x.strip()]
        if not records:
            raise ValueError("--input-generations contains no records")
    else:
        records = [{"prompt_index": i, "prompt": text} for i, text in enumerate(texts)]
    generation_path = Path(args.generations_output) if args.generations_output else None
    generation_records = []
    chunk_dir = Path(args.output).with_suffix("").with_name(Path(args.output).stem + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    all_labels_count = 0
    all_green_count = 0
    layers = None
    rows_by_layer = None
    chunk_labels = []
    prompt_in_chunk = 0
    processed = 0

    def flush_chunk(chunk_index):
        nonlocal chunk_labels, rows_by_layer, prompt_in_chunk
        if not chunk_labels:
            return
        labels_np = np.asarray(chunk_labels, dtype=np.int64)
        rows = rows_by_layer
        if args.balance:
            rng = np.random.default_rng(args.seed + chunk_index)
            green = np.flatnonzero(labels_np == 1)
            red = np.flatnonzero(labels_np == 0)
            n = min(len(green), len(red))
            if n == 0:
                raise RuntimeError(f"Chunk {chunk_index} contains only one class; disable --balance or increase chunk-size")
            keep = np.concatenate([rng.choice(green, n, replace=False), rng.choice(red, n, replace=False)])
            rng.shuffle(keep)
            labels_np = labels_np[keep]
            rows = [[row[i] for i in keep] for row in rows]
        path = chunk_dir / f"part-{chunk_index:05d}.pt"
        torch.save({"embeddings": [torch.stack(row) for row in rows],
                    "labels": torch.from_numpy(labels_np)}, path)
        chunk_paths.append(str(path))
        print(f"saved chunk {chunk_index}: {len(labels_np)} rows -> {path}", flush=True)
        chunk_labels = []
        rows_by_layer = [[] for _ in range(layers)]
        prompt_in_chunk = 0

    with torch.inference_mode():
        for index, record in enumerate(records):
            prompt = record["prompt"]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            prompt_ids = enc["input_ids"].to(device)
            prompt_mask = enc["attention_mask"].to(device)
            if args.input_generations:
                continuation_ids = tokenizer(record["continuation"], add_special_tokens=False,
                                             return_tensors="pt")["input_ids"].to(device)
                generated = torch.cat([prompt_ids, continuation_ids], dim=1)
            else:
                generated = model.generate(
                    input_ids=prompt_ids, attention_mask=prompt_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.greedy,
                    temperature=max(args.temperature, 1e-5), top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
                continuation = generated[0, int(prompt_mask[0].sum()):]
                generation_records.append({
                    "prompt_index": index, "prompt": prompt,
                    "continuation": tokenizer.decode(continuation, skip_special_tokens=True),
                    "prompt_tokens": int(prompt_mask[0].sum()),
                    "generated_tokens": int(len(continuation)),
                })
            prompt_len = int(prompt_mask[0].sum())
            ids = generated[0].detach().cpu()
            out = model(input_ids=generated, output_hidden_states=True, return_dict=True)
            if layers is None:
                layers = len(out.hidden_states)
                rows_by_layer = [[] for _ in range(layers)]
            for t in range(max(min_context - 1, prompt_len - 1), len(ids) - 1):
                target = int(ids[t + 1])
                green = detector._get_greenlist_ids(ids[:t + 1])
                chunk_labels.append(int(target in green.tolist()))
                for li, hidden in enumerate(out.hidden_states):
                    rows_by_layer[li].append(hidden[0, t].float().cpu())
                processed += 1
                if args.max_points and processed >= args.max_points:
                    break
            prompt_in_chunk += 1
            all_labels_count += len(ids) - max(min_context - 1, prompt_len - 1) - 1
            # Count raw green from this chunk before balancing.
            all_green_count += sum(chunk_labels[-(len(ids) - max(min_context - 1, prompt_len - 1) - 1):])
            print(f"processed prompts {index + 1}/{len(records)}; points={processed}", flush=True)
            if prompt_in_chunk >= args.chunk_size or (args.max_points and processed >= args.max_points):
                flush_chunk(len(chunk_paths))
            if args.max_points and processed >= args.max_points:
                break
    flush_chunk(len(chunk_paths))
    if generation_path and not args.input_generations:
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in generation_records), encoding="utf-8")
        print(f"saved generations: {generation_path}")
    manifest = {
        "format": "kgw_probe_chunks_v1", "chunks": chunk_paths,
        "metadata": {"model": args.model, "k": args.k, "delta": args.delta,
                     "gamma": args.gamma, "seeding_scheme": scheme,
                     "context_width": detector.context_width, "balanced": args.balance,
                     "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                     "top_p": args.top_p, "raw_count": int(all_labels_count),
                     "raw_green_count": int(all_green_count),
                     "count": None, "layers": layers,
                     "generated_prompt_count": len(records)},
    }
    torch.save(manifest, args.output)
    print(f"saved manifest: {args.output}")


if __name__ == "__main__":
    main()
