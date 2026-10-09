#!/usr/bin/env python
"""Live full-vocabulary KGW logit-lens analysis for base vs k0.

This script does not save full logits. It loads both models, runs inference,
projects selected hidden states through the final norm and lm_head, and keeps
only layer summaries, histogram samples, and plots.

Colab example::

    !python analysis/analyze_kgw_logit_lens_live.py \
      --family llama --model-a base --model-b k0 --delta 2 \
      --tokenizer meta-llama/Llama-2-7b-hf --hf-token "$HF_TOKEN" \
      --max-tokens 512 --hist-samples 200000
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    "Describe how a city can prepare for extreme heat while protecting vulnerable residents.",
    "Give a concise explanation of why scientific experiments need control groups.",
    "The historian opened the archive and discovered that the letter began with",
    "List several ways to make a machine-learning model's predictions easier to evaluate.",
]


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=BASE_REPOS, default="llama")
    p.add_argument("--model-a", default="base")
    p.add_argument("--model-b", default="k0")
    p.add_argument("--delta", type=int, choices=(1, 2), default=2)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--prompt-file")
    p.add_argument("--prompt", action="append", dest="prompts")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--layers", default="all", help="all or comma-separated hidden-state layers")
    p.add_argument("--hist-samples", type=int, default=200000,
                   help="Maximum vocabulary-logit samples per layer/model for histograms")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--kgw-gamma", type=float, default=0.25)
    p.add_argument("--kgw-seeding-scheme", choices=("simple_0", "simple_1", "simple_2"), default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def resolve(value, family, delta):
    if value in ("base", "k0", "k1", "k2"):
        repo = BASE_REPOS[family] if value == "base" else WATERMARK_REPOS[family].format(variant=value, delta=delta)
        local = Path("pretrained") / repo.rsplit("/", 1)[-1]
        return str(local) if (local / "config.json").is_file() else repo
    return value


def get_prompts(ns):
    if ns.prompts:
        return ns.prompts
    if ns.prompt_file:
        return [x.strip() for x in Path(ns.prompt_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    return DEFAULT_PROMPTS


def scheme_for(model, explicit):
    if explicit:
        return explicit
    return {"k0": "simple_0", "k1": "simple_1", "k2": "simple_2"}.get(model, "simple_0")


def make_masks(tokenizer, prompts, vocab_size, gamma, scheme, max_tokens):
    from watermarks.kgw.watermark_processor import WatermarkBase
    encoded = tokenizer(prompts, add_special_tokens=True, padding=False, truncation=False)
    watermark = WatermarkBase(vocab=list(range(vocab_size)), gamma=gamma,
                              seeding_scheme=scheme, device="cpu")
    import torch as torch_local
    special_ids = set()
    if scheme == "simple_1":
        special_ids = {int(x) for x in (tokenizer.eos_token_id, tokenizer.bos_token_id,
                                         tokenizer.pad_token_id, tokenizer.unk_token_id)
                       if x is not None and 0 <= int(x) < vocab_size}
    masks = []
    prompt_index = []
    for prompt_id, ids in enumerate(encoded["input_ids"]):
        ids = np.asarray(ids, dtype=np.int64)
        for position in range(len(ids)):
            context = ids[:position + 1]
            if len(context) < watermark.context_width:
                masks.append(np.zeros(vocab_size, dtype=bool))
            elif scheme == "simple_1" and int(context[-1]) in special_ids:
                masks.append(np.zeros(vocab_size, dtype=bool))
            else:
                green = watermark._get_greenlist_ids(torch_local.as_tensor(context, dtype=torch_local.long)).tolist()
                mask = np.zeros(vocab_size, dtype=bool)
                mask[np.asarray(green, dtype=np.int64)] = True
                if special_ids:
                    mask[list(special_ids)] = False
                masks.append(mask)
            prompt_index.append(prompt_id)
    masks = np.asarray(masks, dtype=bool)
    prompt_index = np.asarray(prompt_index, dtype=np.int64)
    if len(masks) > max_tokens:
        indices = np.linspace(0, len(masks) - 1, max_tokens, dtype=np.int64)
        masks, prompt_index = masks[indices], prompt_index[indices]
    return encoded, masks, prompt_index


def batches(tokenizer, prompts, max_length, batch_size):
    return [tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
            for i in range(0, len(prompts), batch_size)]


def collect_hidden(model_source, encoded_batches, token, device, trust_remote_code, max_tokens):
    from transformers import AutoModelForCausalLM
    kw = {"token": token, "low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        offload = ROOT / "analysis" / "offload" / Path(model_source).name
        offload.mkdir(parents=True, exist_ok=True)
        kw.update(torch_dtype=torch.float16, device_map="auto", offload_folder=str(offload))
    else:
        kw["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_source, **kw)
    model.eval()
    norm_module = next(x for x in [getattr(getattr(model, "model", None), "norm", None),
                                   getattr(getattr(model, "transformer", None), "ln_f", None),
                                   getattr(getattr(model, "gpt_neox", None), "final_layer_norm", None)] if x is not None)
    head_module = model.get_output_embeddings()
    captured = {"norm_weight": None, "norm_bias": None, "head_weight": None, "head_bias": None}

    def read_real_parameter(module, parameter_name):
        parameter = getattr(module, parameter_name, None)
        if parameter is not None and not parameter.is_meta:
            return parameter.detach().float().cpu().clone()
        # Accelerate disk offload keeps the actual checkpoint tensor in
        # weights_map while module parameters are meta placeholders.
        hook = getattr(module, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is not None:
            try:
                value = weights_map[parameter_name]
                if isinstance(value, torch.Tensor) and not value.is_meta:
                    return value.detach().float().cpu().clone()
            except (KeyError, TypeError, AttributeError):
                pass
        return None

    def capture_norm(module, inputs):
        captured["norm_weight"] = read_real_parameter(module, "weight")
        captured["norm_bias"] = read_real_parameter(module, "bias")

    def capture_head(module, inputs):
        if captured["head_weight"] is None:
            captured["head_weight"] = read_real_parameter(module, "weight")
            captured["head_bias"] = read_real_parameter(module, "bias")

    norm_hook = norm_module.register_forward_pre_hook(capture_norm)
    head_hook = head_module.register_forward_pre_hook(capture_head)
    chunks = None
    with torch.inference_mode():
        for batch in encoded_batches:
            input_device = model.get_input_embeddings().weight.device
            inputs = {k: v.to(input_device) for k, v in batch.items()}
            output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            valid = inputs["attention_mask"].bool()
            if chunks is None:
                chunks = [[] for _ in output.hidden_states]
            for layer, state in enumerate(output.hidden_states):
                chunks[layer].append(state[valid].float().cpu())
            del output, inputs
    norm_hook.remove()
    head_hook.remove()
    if chunks is None:
        raise ValueError("No hidden states collected")
    if captured["norm_weight"] is None or captured["head_weight"] is None:
        raise RuntimeError("Could not capture final norm/lm_head weights during model forward")
    model._logit_lens_projection_weights = captured
    hidden = [torch.cat(x).numpy() for x in chunks]
    if hidden[0].shape[0] > max_tokens:
        indices = np.linspace(0, hidden[0].shape[0] - 1, max_tokens, dtype=np.int64)
        hidden = [x[indices] for x in hidden]
    return model, hidden


def selected_layers(spec, count):
    if spec == "all":
        return list(range(count))
    result = sorted({int(x.strip()) for x in spec.split(",")})
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"--layers must be within 0..{count - 1}")
    return result


def project_model(model, hidden, layers, masks, hist_samples, bins, rng):
    import torch.nn.functional as F
    weights = model._logit_lens_projection_weights
    norm_weight = weights["norm_weight"]
    norm_bias = weights["norm_bias"]
    head_weight = weights["head_weight"]
    head_bias = weights["head_bias"]
    config = model.config
    norm_type = getattr(config, "rms_norm_eps", None)
    eps = float(norm_type if norm_type is not None else getattr(config, "layer_norm_eps", 1e-5))
    vocab = int(head_weight.shape[0])
    rows = []
    samples = {}
    for layer in layers:
        with torch.inference_mode():
            h = torch.from_numpy(hidden[layer]).float()
            nw = norm_weight.to(dtype=torch.float32)
            if norm_bias is None:
                # Llama RMSNorm: divide by RMS, then multiply learned scale.
                normalized = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * nw
            else:
                # LayerNorm-style final normalization (e.g. GPT-NeoX).
                normalized = F.layer_norm(h, (h.shape[-1],), nw, norm_bias, eps)
            projected = F.linear(normalized, head_weight, head_bias)
            values = projected.detach().cpu().numpy()
        flat = values.reshape(-1)
        flat_mask = np.broadcast_to(masks, values.shape)
        green_values = values[flat_mask]
        red_values = values[~flat_mask]
        # Exact group means/counts; only sampled values are used for histograms.
        rows.append({"layer": layer, "green_mean": float(green_values.mean()),
                     "red_mean": float(red_values.mean()),
                     "green_red_gap": float(green_values.mean() - red_values.mean()),
                     "green_count": int(green_values.size), "red_count": int(red_values.size),
                     "all_mean": float(flat.mean()), "all_std": float(flat.std())})
        if flat.size > hist_samples:
            pick = rng.choice(flat.size, hist_samples, replace=False)
        else:
            pick = np.arange(flat.size)
        samples[layer] = (flat[pick], flat_mask.reshape(-1)[pick])
        del h, projected, values
    return rows, samples, vocab


def plot_all(out, results, sample_sets, masks, layers, model_names, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#2563eb", "k0": "#f97316", "k1": "#f97316"}
    # Figure 1: exact green/red mean and gap versus layer.
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in model_names:
        rs = results[name]
        x = [r["layer"] for r in rs]
        ax.plot(x, [r["green_mean"] for r in rs], color="#16a34a", linestyle="-",
                marker="o", label=f"{name} green mean")
        ax.plot(x, [r["red_mean"] for r in rs], color="#dc2626", linestyle="--",
                marker="o", label=f"{name} red mean")
    ax.set_title("Full-vocabulary KGW green/red logit-lens means")
    ax.set_xlabel("hidden-state layer"); ax.set_ylabel("projected logit")
    ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout(); means_path = out / "full_vocab_green_red_means.png"
    fig.savefig(means_path, dpi=170, bbox_inches="tight"); plt.close(fig)

    # Figure 2: each selected layer has base/k0 panels; each panel overlays
    # all-logit, green-mask, and red-mask distributions with mean lines.
    fig, axes = plt.subplots(len(layers), len(model_names), figsize=(15, max(5, 3.4 * len(layers))),
                             squeeze=False, sharex="row")
    for row, layer in enumerate(layers):
        for col, name in enumerate(model_names):
            ax = axes[row, col]
            vals, mask = sample_sets[name][layer]
            ax.hist(vals, bins=bins, alpha=.28, color="#64748b", density=True, label="all")
            if np.any(mask):
                ax.hist(vals[mask], bins=bins, alpha=.45, color="#16a34a", density=True, label="green")
            if np.any(~mask):
                ax.hist(vals[~mask], bins=bins, alpha=.45, color="#dc2626", density=True, label="red")
            result = next(r for r in results[name] if r["layer"] == layer)
            ax.axvline(result["green_mean"], color="#16a34a", linewidth=1.8)
            ax.axvline(result["red_mean"], color="#dc2626", linestyle="--", linewidth=1.8)
            ax.set_title(f"{name}, layer {layer}")
            ax.grid(alpha=.2)
            if row == 0 and col == 0: ax.legend(fontsize=8)
    fig.suptitle("Full-vocabulary logit-lens distributions with KGW masks")
    fig.tight_layout(); hist_path = out / "full_vocab_layer_distributions.png"
    fig.savefig(hist_path, dpi=170, bbox_inches="tight"); plt.close(fig)
    return means_path, hist_path


def main():
    ns = args()
    prompts = get_prompts(ns)
    token = ns.hf_token or os.environ.get("HF_TOKEN")
    source_a, source_b = resolve(ns.model_a, ns.family, ns.delta), resolve(ns.model_b, ns.family, ns.delta)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ns.tokenizer, token=token, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    encoded_raw, masks, prompt_index = make_masks(tokenizer, prompts, 32000, ns.kgw_gamma,
                                                   scheme_for(ns.model_b, ns.kgw_seeding_scheme), ns.max_tokens)
    batches_a = batches(tokenizer, prompts, ns.max_length, ns.batch_size)
    # Both models must use exactly the same tokenized batches.
    for batch in batches_a:
        if batch["input_ids"].numel() == 0: raise ValueError("Empty tokenized batch")
    device = torch.device("cuda" if ns.device == "auto" and torch.cuda.is_available() else "cpu" if ns.device == "auto" else ns.device)
    print(f"Live logit-lens: base={source_a}; watermarked={source_b}; device={device}; prompts={len(prompts)}")
    model_a, hidden_a = collect_hidden(source_a, batches_a, token, device, ns.trust_remote_code, ns.max_tokens)
    model_b, hidden_b = collect_hidden(source_b, batches_a, token, device, ns.trust_remote_code, ns.max_tokens)
    if hidden_a[0].shape[0] != masks.shape[0]:
        raise ValueError(f"Mask/token mismatch: masks={masks.shape[0]}, hidden={hidden_a[0].shape[0]}")
    layers = selected_layers(ns.layers, len(hidden_a))
    out = Path(ns.output_dir or f"analysis/live_kgw_logit_lens_{ns.model_a}_vs_{ns.model_b}_delta{ns.delta}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    results_a, samples_a, vocab_a = project_model(model_a, hidden_a, layers, masks, ns.hist_samples, ns.bins, rng)
    del model_a, hidden_a; gc.collect();
    if device.type == "cuda": torch.cuda.empty_cache()
    results_b, samples_b, vocab_b = project_model(model_b, hidden_b, layers, masks, ns.hist_samples, ns.bins, rng)
    del model_b, hidden_b; gc.collect();
    results = {"base": results_a, "k0": results_b}
    sample_sets = {"base": samples_a, "k0": samples_b}
    with (out / "full_vocab_group_summary.csv").open("w", encoding="utf-8", newline="") as f:
        import csv
        fields = ["model", *results_a[0].keys()]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, rows in results.items():
            for row in rows: w.writerow({"model": name, **row})
    paths = plot_all(out, results, sample_sets, masks, layers, ("base", "k0"), ns.bins)
    (out / "metadata.json").write_text(json.dumps({"prompts": prompts, "layers": layers,
        "vocab_size": vocab_a, "kgw_gamma": ns.kgw_gamma,
        "kgw_seeding_scheme": scheme_for(ns.model_b, ns.kgw_seeding_scheme),
        "hist_samples": ns.hist_samples, "bins": ns.bins,
        "definition": "full vocabulary projected logits; histogram values sampled, means exact"}, indent=2), encoding="utf-8")
    print(f"Saved summary: {out / 'full_vocab_group_summary.csv'}")
    for path in paths: print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
