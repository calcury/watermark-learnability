# On the Learnability of Watermarks for Language Models

This repository contains code for the ICLR 2024 paper [On the Learnability of Watermarks for Language Models](https://arxiv.org/abs/2312.04469) by Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto.

### Setup

To install the necessary packages, first create a conda environment.
```shell
conda create -n <env_name> python=3.11
conda activate <env_name>
```
Then, install the required packages with 
```shell
pip install -r requirements.txt
```

### Usage

We include scripts for reproducing experiments in the paper in the [`scripts`](scripts) directory, which also serve as examples for how to run the files in this repository. `README.md`'s within [`scripts`](scripts) provide instructions on how to run the scripts. Note that all scripts should be run from the top-level directory.

Feel free to create an issue if you encounter any problems or bugs!

### Pythia KGW sampling-distillation experiments (alignment loss)

The added utilities compare ordinary sampling distillation against an auxiliary
KGW-direction cosine loss. They support `k=0/1/2` (schemes `simple_0/1/2`) and
`delta=1/2`; the default dataset ID is
`cygu/sampling-distill-train-data-kgw-k{k}-gamma0.25-delta{delta}`.

Fetch the original Pythia base using the existing repository mapping:

```bash
python analysis/fetch_pythia.py base
```

Train the original sampling-distillation baseline (auxiliary loss disabled). The wrapper defaults to a low-memory setup (microbatch 1, block size 128, Adafactor, gradient checkpointing):

```bash
python analysis/train_pythia_sampling.py --k 0 --delta 1 --mode baseline --bf16
```

The original paper-style setting (batch 4, block 256, AdamW, no checkpointing) can exceed a 14.6 GiB GPU. If it still OOMs, reduce `--block-size` (e.g. 64), keep `--batch-size 1`, and use `--optim adafactor`; try a 24+ GiB GPU or parameter-efficient fine-tuning if full Pythia training still does not fit. Raise batch/block size or use `--no-gradient-checkpointing` only if memory allows. To try standard AdamW, pass `--optim adamw_torch` (it needs substantially more memory). The alignment run also loads a frozen reference model; keep `--reference-device cpu` on small GPUs.

Train with the cosine alignment loss:

```bash
python analysis/train_pythia_sampling.py --k 0 --delta 1 --mode align \
  --alignment-loss-weight 0.1 --bf16
```

The additional objective is `L = L_sampling_CE + lambda * (1 - cosine)`, where
cosine compares the centered student-minus-frozen-clean-reference logits with
the context-seeded centered KGW green/red mask. The reference defaults to the
same Pythia initialization and stays on CPU by default to save GPU memory. If
student and reference logits are identical at initialization, that batch's
undefined cosine term is skipped and the ordinary sampling loss still trains
the model. To reduce KGW mask construction overhead, alignment uses up to 8
evenly spaced valid token positions per sequence by default; adjust with
`--alignment-positions-per-sequence`. Outputs default to `analysis/trained/pythia-1.4b-k{k}-delta{delta}-{baseline|align}`.
Useful options include `--dataset`, `--model`, `--reference-model`,
`--reference-device`, `--batch-size`, `--gradient-accumulation-steps`,
`--max-train-samples`, `--learning-rate`, and `--output-dir`.

To generate samples and report a KGW p-value, place held-out prompts (one per
line) in a UTF-8 file and run. This tests whether the model emits a detectable
pattern by itself; no watermark logits processor is applied during generation:

```bash
python analysis/evaluate_kgw_pvalue.py \
  --model analysis/trained/pythia-1.4b-k0-delta1-align \
  --prompts analysis/probe_prompts.txt --output analysis/pvalue_k0_align.jsonl \
  --k 0 --delta 1 --max-new-tokens 200 --device cuda
```

For k=2, delta=1, pass `--dataset` explicitly if you have a matching dataset;
that Hub combination is not assumed to exist.

P-values are computed from generated continuation tokens only; prompt tokens
provide seed context but are not counted. Per-prompt generations and scores are
written to JSONL, with a companion `.summary.json`. Compare multiple held-out
prompts and an unwatermarked control; small p-values alone are not sufficient
to establish improved utility or robust watermark learning.

### References

Code in the [`watermarks/kgw`](watermarks/kgw) directory is from [github.com/jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking). In the [`watermarks/kth`](watermarks/kth) directory, `detect.py`, `levenshtein.pyx`, and `mersenne.py` are from [github.com/jthickstun/watermark](https://github.com/jthickstun/watermark). [`train_logit_distill.py`](train_logit_distill.py) and [`train_sampling_distill.py`](train_sampling_distill.py) are adapted from [github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py).

## Models

Below are links to trained model weights from the paper's experiments (hosted on Hugging Face). They can also be found at this [Hugging Face collection](https://huggingface.co/collections/cygu/on-the-learnability-of-watermarks-for-language-models-663b6f7e077aba104d461497).

### Logit-based watermark distilled Llama 2 7B

- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift256)

### Sampling-based watermark distilled Llama 2 7B

- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift256)

### Sampling-based watermark distilled Pythia 1.4B

- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift256)

## Training data for sampling-based watermark distillation

Below are links to the watermarked training data used for the paper's sampling-based watermark distillation experiments (hosted on Hugging Face). They can also be found at this [Hugging Face collection](https://huggingface.co/collections/cygu/on-the-learnability-of-watermarks-for-language-models-663b6f7e077aba104d461497).

- [KGW ](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/datasets/cygu/sampling-distill-train-data-aar-k2)
- [Aar k = 3](https://huggingface.co/datasets/cygu/sampling-distill-train-data-aar-k3)
- [Aar k = 4](https://huggingface.co/datasets/cygu/sampling-distill-train-data-aar-k4)
- [KTH s = 1](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kth-shift1)
- [KTH s = 2](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kth-shift2)
- [KTH s = 4](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kth-shift4)
- [KTH s = 256](https://huggingface.co/datasets/cygu/sampling-distill-train-data-kth-shift256)

## Citation

Please cite this paper using the following BibTex entry:
```bibtex
@inproceedings{gu2024learnability,
    title={On the Learnability of Watermarks for Language Models},
    author={Chenchen Gu and Xiang Lisa Li and Percy Liang and Tatsunori Hashimoto},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=9k0krNzvlV}
}
```
