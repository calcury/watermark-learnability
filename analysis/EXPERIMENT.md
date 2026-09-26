# A1/A2 -> B1/B2 watermark representation experiment

## Causal setup

Use the same architecture, tokenizer, training data, optimizer, learning-rate
schedule, steps, batch order, student initialization, and random seeds. The
only intended difference is whether the teacher's output distribution contains
the KGW watermark:

- **A1**: normal teacher
- **A2**: watermarked teacher
- **B1**: student distilled from A1
- **B2**: student distilled from A2

The primary estimand is the paired difference `B2 - B1` on exactly the same
input token IDs. The existing public Pythia checkpoint is a B2-like model and
its base `EleutherAI/pythia-1.4b` is a useful A1/B1 control, but it is **not**
a complete causal reproduction of all four cells. A true reproduction needs
training A1/A2 and then distilling each with the same script and seed.

## Colab reproduction

```python
!pip install -q -U transformers datasets accelerate matplotlib
!python analysis/fetch_model.py
# B1 = pretrained/pythia-1.4b; B2 = the downloaded watermark checkpoint
!python analysis/paired_diff.py \
  --b1 pretrained/pythia-1.4b \
  --b2 pretrained/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2 \
  --prompt-file data/probe.tsv \
  --output-dir analysis/paired_diff_output
```

`data/probe.tsv` can be either one prompt per line or `group<TAB>prompt`.
Pre-register groups such as `clean`, `trigger`, and `random`; do not select a
layer after inspecting the results. The script checks B1/B2 token IDs and
attention masks are identical, loads models sequentially, and reports for every
layer and group:

- per-token cosine distance mean/std/median/P90 and bootstrap 95% CI;
- relative L2 distance and RMS scale;
- linear CKA;
- token count and plots in `paired_diff_output/`.

Use at least 3--5 paired seeds and hundreds/thousands of probe tokens for a
claim. A same-model/different-seed B1 null is needed to distinguish ordinary
fine-tuning drift from watermark-specific drift.

## Behaviour and causality checks

Representation distance alone does not prove a watermark is present. Pair the
white-box measurements with KGW detection on generated text, clean perplexity,
random-trigger false positives, and held-out trigger templates. For stronger
evidence, activation-patch a pre-registered layer from B2 into B1 and test
whether watermark detection changes while clean utility remains stable.

The public model's output watermark can be checked with the repository's
existing KGW detector (`experiments/compute_metrics.py` and
`watermarks/kgw/watermark_processor.py`). Use the same generation settings
and watermark key/config for all four cells.

## Removal experiments (planned interfaces)

Removal must be evaluated as a Pareto problem: watermark detection/ASR should
decrease while clean utility and random-trigger false-positive rate remain
stable. Recommended pre-registered sequence:

1. **R0 null**: no edit and a random weight edit control.
2. **R1 black-box**: clean-only continued training of B2, scanning a fixed
   number of steps and learning rates.
3. **R2 grey-box**: clean-only KD from A1 or the original base model.
4. **R3 white-box**: activation patching or projection at the pre-registered
   layer, evaluated on unseen trigger templates.
5. **R4 upper bound**: trigger-aware unlearning; report it explicitly as an
   informed upper bound, not a general removal method.

Every edit must save a new model directory, parameter-change norm, clean loss,
watermark detection score, random-trigger score, and held-out score. A drop in
watermark detection alone is not success if the model's language quality has
also collapsed.
