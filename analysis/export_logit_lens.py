#!/usr/bin/env python
"""Export a practical logit-lens representation for two causal LMs.

Each hidden state is passed through the model's final normalization and output
head. The exporter saves top-k projected vocabulary logits rather than the full
layer x token x vocabulary tensor, which keeps Llama outputs manageable.

Example::

    python analysis/export_logit_lens.py --family llama --model-a base --model-b k0 --delta 2
    python analysis/export_logit_lens.py --model-a /models/base --model-b /models/k0
"""

import argparse
import gc
import getpass
import json
import os
from pathlib import Path

import numpy as np
import torch

BASE_REPOS = {"llama": "meta-llama/Llama-2-7b-hf", "pythia": "EleutherAI/pythia-1.4b"}
WATERMARK_REPOS = {
    "llama": "cygu/llama-2-7b-logit-watermark-distill-kgw-{variant}-gamma0.25-delta{delta}",
    "pythia": "cygu/pythia-1.4b-sampling-watermark-distill-kgw-{variant}-gamma0.25-delta{delta}",
}
DEFAULT_PROMPTS = [
    "Explain why the seasons change on Earth in a short paragraph.",
    "A careful scientist records uncertainty instead of hiding it.",
    "Write three practical suggestions for reducing household energy use.",
    "The old railway station stood at the edge of the town, where",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=BASE_REPOS, default="llama")
    p.add_argument("--model-a", default="base", help="base/k0/k1/k2, local path, or Hub ID")
    p.add_argument("--model-b", default="k0", help="base/k0/k1/k2, local path, or Hub ID")
    p.add_argument("--delta", type=int, choices=(1, 2), default=2)
    p.add_argument("--prompt", dest="prompts", action="append")
    p.add_argument("--prompt-file")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Valid token positions retained per model (default: 512)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Top projected vocabulary logits saved per layer/token")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def resolve(value, family, delta):
    if value in ("base", "k0", "k1", "k2"):
        if value == "base":
            repo = BASE_REPOS[family]
        else:
            if delta == 1 and value == "k2":
                raise ValueError("No default k2-delta1 repository is configured; pass an explicit model path/ID")
            repo = WATERMARK_REPOS[family].format(variant=value, delta=delta)
        local = Path("pretrained") / repo.rsplit("/", 1)[-1]
        return str(local) if (local / "config.json").is_file() else repo
    return value


def prompts(args):
    if args.prompts:
        out = args.prompts
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            out = [line.strip() for line in f if line.strip()]
    else:
        out = DEFAULT_PROMPTS
    if not out:
        raise ValueError("No prompts supplied")
    return out


def tokenizer(source, token, trust_remote_code):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source, token=token, use_fast=True,
                                        trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def batches(tok, texts, max_length, batch_size):
    return [tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=max_length)
            for i in range(0, len(texts), batch_size)]


def final_norm(model):
    candidates = [
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "gpt_neox", None), "final_layer_norm", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("Could not locate the model's final normalization layer")


def model_kwargs(source, token, device, trust_remote_code):
    kwargs = {"token": token, "low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        offload = Path("analysis/offload") / Path(source).name
        offload.mkdir(parents=True, exist_ok=True)
        kwargs.update(torch_dtype=torch.float16, device_map="auto", offload_folder=str(offload))
    else:
        kwargs["torch_dtype"] = torch.float32
    return kwargs


def collect_lens(source, encoded, token, device, trust_remote_code, top_k, max_tokens):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        source, **model_kwargs(source, token, device, trust_remote_code))
    model.eval()
    norm = final_norm(model)
    head = model.get_output_embeddings()
    if head is None:
        raise RuntimeError("Model has no output embedding/lm_head")
    hidden_chunks = None
    id_chunks = []
    with torch.inference_mode():
        for batch in encoded:
            input_device = model.get_input_embeddings().weight.device
            inputs = {key: value.to(input_device) for key, value in batch.items()}
            output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            valid = inputs["attention_mask"].bool()
            if hidden_chunks is None:
                hidden_chunks = [[] for _ in output.hidden_states]
            for layer, state in enumerate(output.hidden_states):
                hidden_chunks[layer].append(state[valid].float().cpu())
            id_chunks.append(inputs["input_ids"][valid].cpu())
            del output, inputs
    ids = torch.cat(id_chunks).numpy()
    hidden = [torch.cat(chunks).numpy() for chunks in hidden_chunks]
    if ids.shape[0] > max_tokens:
        indices = np.linspace(0, ids.shape[0] - 1, max_tokens, dtype=np.int64)
        ids = ids[indices]
        hidden = [state[indices] for state in hidden]
    top_ids = np.empty((layers, len(ids), top_k), dtype=np.int32)
    top_values = np.empty((layers, len(ids), top_k), dtype=np.float32)
    direction_cosine = np.empty((layers, len(ids)), dtype=np.float32)
    norm_device = next(norm.parameters()).device
    head_device = next(head.parameters()).device
    final = norm(torch.from_numpy(hidden[-1]).to(norm_device)).float()
    final = final.cpu().numpy()
    for layer, state in enumerate(hidden):
        normalized = norm(torch.from_numpy(state).to(norm_device)).float()
        projected = head(normalized.to(head_device)).float()
        values, indices = torch.topk(projected, k=min(top_k, projected.shape[-1]), dim=-1)
        k = values.shape[-1]
        top_values[layer, :, :k] = values.cpu().numpy()
        top_ids[layer, :, :k] = indices.cpu().numpy()
        if k < top_k:
            top_values[layer, :, k:] = np.nan
            top_ids[layer, :, k:] = -1
        a = normalized.cpu().numpy()
        an = np.linalg.norm(a, axis=1).clip(min=1e-12)
        fn = np.linalg.norm(final, axis=1).clip(min=1e-12)
        direction_cosine[layer] = np.sum(a * final, axis=1) / (an * fn)
        del normalized, projected
    info = {"source": source, "hidden_state_count": layers,
            "hidden_size": int(hidden[0].shape[1]), "vocab_size": int(head.out_features)}
    del model, norm, head, hidden
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return top_ids, top_values, direction_cosine, ids, info


def main():
    args = parse_args()
    texts = prompts(args)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    source_a = resolve(args.model_a, args.family, args.delta)
    source_b = resolve(args.model_b, args.family, args.delta)
    if any("meta-llama/" in source for source in (source_a, source_b)) and not token:
        token = getpass.getpass("Hugging Face token for gated Llama model (input hidden): ")
        if not token:
            raise ValueError("A Hugging Face token is required")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tok_a = tokenizer(source_a, token, args.trust_remote_code)
    tok_b = tokenizer(source_b, token, args.trust_remote_code)
    encoded_a = batches(tok_a, texts, args.max_length, args.batch_size)
    encoded_b = batches(tok_b, texts, args.max_length, args.batch_size)
    for index, (a, b) in enumerate(zip(encoded_a, encoded_b)):
        if not torch.equal(a["input_ids"], b["input_ids"]) or not torch.equal(a["attention_mask"], b["attention_mask"]):
            raise ValueError(f"Tokenizers differ in batch {index}; logit-lens positions are not aligned")
    del tok_b, encoded_b
    print(f"Exporting logit lens: A={source_a}; B={source_b}; delta={args.delta}; device={device}")
    result_a = collect_lens(source_a, encoded_a, token, device, args.trust_remote_code, args.top_k, args.max_tokens)
    result_b = collect_lens(source_b, encoded_a, token, device, args.trust_remote_code, args.top_k, args.max_tokens)
    if result_a[3].shape != result_b[3].shape or not np.array_equal(result_a[3], result_b[3]):
        raise ValueError("Models produced different retained token IDs")
    safe_a = Path(args.model_a).name.replace("/", "_")
    safe_b = Path(args.model_b).name.replace("/", "_")
    out = Path(args.output_dir or f"analysis/logit_lens_{safe_a}_vs_{safe_b}_delta{args.delta}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "top_token_ids.npy", np.stack([result_a[0], result_b[0]]))
    np.save(out / "top_logits.npy", np.stack([result_a[1], result_b[1]]))
    np.save(out / "direction_cosine.npy", np.stack([result_a[2], result_b[2]]))
    np.save(out / "token_ids.npy", result_a[3])
    metadata = {"family": args.family, "delta": args.delta, "model_a": result_a[4], "model_b": result_b[4],
                "prompts": texts, "top_k": args.top_k, "max_tokens": args.max_tokens,
                "definition": "final_norm(hidden_state[layer]) projected through lm_head",
                "direction_cosine": "cosine(final_norm(hidden_state[layer]), final_norm(hidden_state[last]))"}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved logit-lens NPY results to {out}")


if __name__ == "__main__":
    main()
