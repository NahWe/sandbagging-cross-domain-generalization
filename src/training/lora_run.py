"""Entry point that actually needs a GPU: loads Mistral-7B-Instruct, LoRA
password-locks it (or trains the matched control) on domain A, evaluates the
held-out split under both framings, and writes an ItemResult-compatible JSON
so results plug straight into src/analysis/ with no glue code.

Adapted from TeunvdWeij/sandbagging's src/training/train.py (load_model,
train_model) and src/wmdp_sandbagging/train_pw_locked_sandbagger.py (the
A/B/C/D forced-choice evaluation trick), scoped to a single --condition run
instead of always training the sandbagging intervention -- see
scripts/run_all_seeds.sh for the seed x condition sweep this is meant to be
called from.

NOT unit-tested here on purpose (see docs/design.md's "Status": no GPU runs
yet). The dataset construction this depends on (data_prep.py) IS unit-tested
with dummy data -- this file is intentionally the one place in the repo that
needs a real model to verify, and is kept as thin as possible around that
dependency so there's as little untested logic here as possible.
"""
import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Quiets the per-parameter "Loading weights: X%|... Materializing param=..."
# tqdm bar (291 update lines per model load) and library warnings
# (torch_dtype deprecation, use_cache/gradient-checkpointing notice) --
# real signal is the training-progress prints below, not this noise.
#
# try/except: got this wrong once already (transformers.utils
# .disable_progress_bar doesn't exist -- it's nested under
# transformers.utils.logging, confirmed by inspecting the installed
# package directly instead of trusting docs/search results a second
# time), which crashed every training run at import time. Cosmetic log
# suppression must never be able to take down training again regardless
# of which exact API a given transformers version exposes.
try:
    transformers.utils.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except AttributeError:
    pass

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import (
    PASSWORD,
    build_control_examples,
    build_distillation_locked_examples,
    build_format_control_examples,
    build_format_eval_examples,
    build_format_locked_examples,
    build_locked_examples,
    split_items,
    write_jsonl,
)
from src.training.format_triggers import HELDOUT_WRAPPER_PAIRS
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS as FORMAT_TRAIN_WRAPPER_PAIRS
from src.training.semantic_triggers import (
    EXTRAPOLATION_HELDOUT_PAIRS as SEMANTIC_EXTRAPOLATION_HELDOUT_PAIRS,
    INTERPOLATION_HELDOUT_PAIRS as SEMANTIC_INTERPOLATION_HELDOUT_PAIRS,
    TRAIN_WRAPPER_PAIRS as SEMANTIC_TRAIN_WRAPPER_PAIRS,
)
from src.training.weak_reference import DEFAULT_WEAK_MODEL, compute_weak_distributions, weak_model_accuracy

HF_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

# From TeunvdWeij/sandbagging's train_and_evaluate_mistral_for_wmdp_sandbagging.sh
# -- docs/design.md, "Password-locking (metodo)".
LR = 5e-6
# BATCH_SIZE x GRAD_ACCUM_STEPS = 16, preserving the reference repo's
# effective batch size exactly. 8x2 matches the reference's own split
# exactly -- restored now that 4-bit quantization confirmed live (~4GB
# for the base model, of a 16GB card) leaves real headroom; the earlier
# 4x4 split was a memory-safety measure from when the unquantized model
# alone used ~14.5GB.
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 3

# Restored to evaluate_framing's original default now that quantization
# leaves headroom -- the 4x4/EVAL_BATCH_SIZE=4 drop was sized against the
# unquantized model using ~14.5GB of a 16GB card, not a properly-measured
# limit on the quantized (~4GB) one.
EVAL_BATCH_SIZE = 16

# Every 2 was for the smoke test, when the open question was "does this
# loop even work" and fast feedback mattered more than a clean log. Now
# that training is confirmed working (measured ~19s/step live), the sweep
# runs for hours per seed/condition -- every 25 steps is roughly one print
# every ~8 minutes at that pace, enough to see it's alive without flooding
# the log across a multi-hour, multi-process (NUM_GPUS>1) run.
TRAIN_PROGRESS_EVERY = 25

# LoRA config confirmed from the reference repo's actual source
# (src/training/train.py), not just the README -- docs/design.md,
# "confirmado 2026-08-04, leyendo el codigo fuente real".
LORA_TARGET_MODULES = [
    "k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    # docs/extension-plan-format-triggered-lock.md: "format" swaps the
    # password token for a randomly-assigned eval/deploy prompt wrapper per
    # item (format_triggers.py) -- everything downstream of dataset
    # construction (model, training loop, evaluate_framing, analysis
    # pipeline) is unchanged either way.
    # "semantic" (docs/discussion-next-experiments-design.md, "1. Trigger
    # semantico con parafrasis") is the same idea at 168 wrapper pairs
    # instead of 16, with a two-tier holdout (semantic_triggers.py) so
    # within-domain generalization gets checked separately for new phrasings
    # of a seen style (interpolation) vs. an entire unseen style
    # (extrapolation) -- format's single held-out set conflated the two.
    parser.add_argument("--trigger", choices=["password", "format", "semantic"], default="password")
    # 2026-08-29, "simpler-than-password" test: format's normal pool draws
    # randomly from 12 trained wrapper pairs per item (cardinality=12) --
    # password is cardinality=1 (always the same fixed token). This isolates
    # trigger CARDINALITY from trigger RECOGNITION COMPLEXITY (semantic vs.
    # lexical) as the reason format transfers less than password: restricts
    # training (and, in evaluate_domain_b.py/evaluate_domain_c.py, the
    # cross-domain eval) to a single fixed wrapper pair, matching password's
    # cardinality while keeping the semantic (non-lexical) trigger property.
    # Only meaningful with --trigger format; ignored for password/semantic.
    parser.add_argument("--trigger-cardinality", choices=["full", "single"], default="full")
    # docs/extension-plan-auditing-games-induction.md, "Option A": varies
    # only what eval-framing trains toward -- a randomly sampled wrong
    # letter (label_flip) vs. a weak reference model's own forced-choice
    # distribution (distillation). Orthogonal to --trigger (composes with
    # password/format/semantic -- main()'s mechanism==distillation branch
    # picks the matching wrapper pool, if any), same as label_flip already
    # is; see docs/discussion-next-experiments-design.md for why the
    # format+distillation combination was worth filling in.
    parser.add_argument("--mechanism", choices=["label_flip", "distillation"], default="label_flip")
    parser.add_argument(
        "--weak-model",
        type=str,
        default=DEFAULT_WEAK_MODEL,
        help="Reference model for --mechanism distillation. Any genuinely "
        "weaker instruct model works -- no tokenizer/family constraint, "
        "see weak_reference.py's module docstring for why.",
    )
    # Both only apply to --mechanism distillation; a no-op default (1.0)
    # reproduces the original, unweighted/untempered behavior exactly.
    # Added after the first real sweep (2026-08-26) showed a much weaker,
    # noisier domain-A gap than label_flip's, and a weak-agreement
    # diagnostic that came out negative on domain B (locked matches the
    # weak model's own wrong answers LESS than control does) -- see
    # docs/status-distillation-mechanism.md for the read. Two cheap knobs
    # to try before escalating to a heavier full-sequence redesign.
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=1.0,
        help="Sharpens (<1) or flattens (>1) the weak model's own "
        "forced-choice distribution before it becomes the soft KL target. "
        "Doesn't change the weak model's argmax (weak_model_accuracy_on_domain_a "
        "is unaffected), only how decisively the KL loss has to commit to "
        "the weak model's specific per-item preference.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=1.0,
        help="Multiplies the soft (KL) loss term relative to the hard "
        "cross-entropy term in train_one_epoch_pass, before both are "
        "averaged into the batch loss. >1 pushes eval-framing training "
        "harder toward imitating the weak model.",
    )
    # Defaults to the real EPOCHS used by the sweep -- only override for a
    # time-boxed smoke test (nothing is checkpointed until this run's very
    # end, so a shorter run is the only way to fit inside a short session).
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def select_dtype(device: str) -> torch.dtype:
    """bf16 needs Ampere or newer (compute capability >= 8.0); older cards
    (Pascal, Turing) either error out or run bf16 unsupported in hardware.
    Fall back to fp16 there -- paired with GradScaler in
    train_one_epoch_pass, since fp16's narrower exponent range (vs. bf16)
    can underflow small gradients to zero without loss scaling."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


# Mistral-7B in fp16/bf16 alone needs ~14.5GB for the frozen base weights.
# 24GB is a conservative cutoff above that floor -- enough real margin for
# LoRA/optimizer/activation overhead (even with gradient checkpointing) on
# top of the base weights. Cards below this genuinely need 4-bit
# quantization to fit at all (observed live: repeated CUDA OOMs on a 16GB
# card at fp16); cards at or above it don't need the approximation and get
# full-precision weights instead.
MIN_VRAM_BYTES_FOR_UNQUANTIZED = 24 * 1024**3


def needs_quantization(device: str) -> bool:
    """Whether to load the frozen base model in 4-bit (QLoRA-style)
    instead of full dtype precision -- decided from the actual GPU's VRAM
    at runtime, not hardcoded to always-on, so this adapts automatically
    across hardware (a 16GB Kaggle T4 vs. a 48GB RunPod A40) instead of
    quantizing everywhere regardless of whether it's needed."""
    if device != "cuda":
        return False
    total_vram = torch.cuda.get_device_properties(0).total_memory
    return total_vram < MIN_VRAM_BYTES_FOR_UNQUANTIZED


def model_kwargs(dtype: torch.dtype, device: str) -> dict:
    """attn_implementation="sdpa" instead of flash-attn: sdpa ships inside
    torch itself, so it needs no CUDA-toolkit compile step -- flash-attn
    wheels are ABI-pinned to a specific torch/CUDA/python combo and aren't
    guaranteed to have a matching build available on every host.

    4-bit quantization (QLoRA-style) only when needs_quantization(device)
    says the card's VRAM actually requires it (see that function). NF4 +
    double quantization shrinks the frozen base weights to ~4GB when
    active. lm_head is excluded from quantization (llm_int8_skip_modules)
    so evaluate_framing's forced A/B/C/D logit read stays full precision
    regardless -- it's a tiny fraction of total size, not worth quantizing
    for the memory it'd save.

    device_map={"": 0} is used in both the quantized and unquantized cases
    here (not just because bitsandbytes requires it for quantized
    tensors) -- it places the whole model on GPU 0 directly while loading,
    same as an explicit .to(device) would for the unquantized path, so one
    code path covers both."""
    if device != "cuda":
        return dict(attn_implementation="sdpa", torch_dtype=dtype)
    kwargs = dict(attn_implementation="sdpa", torch_dtype=dtype, device_map={"": 0})
    if needs_quantization(device):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        )
    return kwargs


def load_model(seed: int, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    # make_collate builds hard-label batches right-padded by construction
    # (pad_id fills the tail after each example's real tokens), and
    # evaluate_framing's last_token_idx = attn_mask.sum(dim=1) - 1 assumes
    # the same -- both only line up with the tokenizer's OWN padding if it
    # also pads right. Confirmed live 2026-08-25 that Mistral-7B-Instruct-
    # v0.2's tokenizer already defaults to "right", so this was previously
    # relying on that default silently; setting it explicitly here means a
    # future HF_PATH swap to a tokenizer defaulting to "left" fails loudly
    # via wrong eval numbers being caught in review, not silently via a
    # library default nobody pinned.
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        # Reuse eos_token instead of adding a new special token + resizing:
        # resize_token_embeddings needs real float weights to compute its
        # mean/covariance init, but bitsandbytes' 4-bit Linear layers store
        # packed Params4bit tensors, not plain floats -- resizing a
        # quantized lm_head isn't a reliable operation when
        # needs_quantization(device) is True. Reusing an existing token
        # sidesteps resizing (and the vocab-size mismatch it would
        # otherwise require) entirely, and is just as fine a default when
        # the base model isn't quantized.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if needs_quantization(device):
        # prepare_model_for_kbit_training's per-parameter pass casts every
        # fp16/bf16 parameter to fp32 for training stability. Cheap when
        # actually quantized -- it skips Params4bit tensors, only
        # upcasting the few remaining fp16/bf16 modules like LayerNorm --
        # but confirmed live 2026-08-25 to genuinely double memory when
        # called on an UNQUANTIZED model: 29.1GB allocated right after
        # load on an A40, vs the ~14.5GB bf16 weights alone should need --
        # almost exactly fp32's 2x, and the direct cause of an OOM on a
        # 44GB card that should have had headroom to spare. Only call this
        # on the quantized path; the else branch below gets the same
        # gradient-checkpointing setup without the unwanted upcast.
        #
        # use_gradient_checkpointing=True: reverted from False (see git
        # history) after hitting a live CUDA OOM during training with it
        # off on a 16GB card. Turing (the T4 this was first observed on)
        # lacks the fast 4-bit GEMM kernel, so a quantized base still
        # dequantizes each Linear4bit layer's full weight matrix to fp16
        # per matmul -- a real per-layer memory cost during backward that
        # checkpointing's activation savings offset.
        #
        # use_reentrant=False: re-enabling checkpointing above didn't
        # actually lower memory use in a live run (still ~14.4GB, same as
        # with it off) -- PyTorch's legacy "reentrant" checkpointing (the
        # default) has known correctness/composability gaps with custom
        # autograd Functions like bitsandbytes' MatMul4Bit, and can fail
        # to release memory the way non-reentrant checkpointing does.
        # Non-reentrant is also the currently-recommended mode in general.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        # Same gradient-checkpointing setup prepare_model_for_kbit_training
        # would do, minus its fp32-upcast-everything step (see above).
        # enable_input_require_grads is still needed here even without
        # quantization: the base model is frozen (LoRA-only training), so
        # without it gradient checkpointing has no requires_grad=True
        # tensor flowing into the checkpointed segment to attach a
        # backward graph to.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    # gradient_checkpointing's HF-side hook only activates in train mode
    # (checked via self.training inside the model's forward) -- set
    # explicitly rather than relying on nn.Module's default.
    model.train()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.prompt, ex.target, ex.soft_target


def build_hard_example_ids(tokenizer, prompt: str, target: str):
    """Builds one hard-label training example's token ids by concatenating
    separately-tokenized prompt/target segments, and returns the index
    where target supervision begins.

    Deliberately NOT built by tokenizing "{prompt} {target}{eos}" as one
    joined string and inferring the prompt/target boundary from a
    separately-tokenized "{prompt} " prefix's length -- SentencePiece
    (Mistral's tokenizer) merges a trailing space into the following
    letter's token only when more text follows it, so a standalone
    "{prompt} " tokenizes to one MORE token (a lone space token) than the
    prompt-prefix actually occupies inside the full joined string (where
    that same space merges into a single "_B"-style token with the
    letter). Confirmed live 2026-08-24 against the real tokenizer: the
    old prompt_len pointed one token past the real boundary, at the eos
    token, not the target letter -- so labels masked out the target
    letter itself and left only eos (a trivial, target-independent
    objective) for the loss to fit. That silently explained why training
    loss converged to ~0 within a fraction of epoch 1, identically for
    both locked and control conditions regardless of which letter was
    supposedly the target: the letter was never actually being
    supervised. Concatenating separately-tokenized id lists sidesteps the
    boundary-inference problem entirely -- the split point is known by
    construction, not inferred after the fact."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(f" {target}{tokenizer.eos_token}", add_special_tokens=False)["input_ids"]
    return prompt_ids + target_ids, len(prompt_ids)


def make_collate(tokenizer):
    """Splits each batch into a hard-label sub-batch (standard next-token
    cross-entropy, exactly as before) and a soft-label sub-batch
    (docs/extension-plan-auditing-games-induction.md's distillation
    mechanism). A batch built entirely from build_locked_examples/
    build_control_examples/build_format_*_examples has target set on every
    row, so its soft sub-batch is empty and it collates exactly as before;
    only build_distillation_locked_examples' eval-framing rows (target=None)
    populate the soft sub-batch."""

    def collate(batch):
        prompts, targets, soft_targets = zip(*batch)
        hard_idx = [i for i, t in enumerate(targets) if t is not None]
        soft_idx = [i for i, t in enumerate(targets) if t is None]

        hard_batch = None
        if hard_idx:
            hard_prompts = [prompts[i] for i in hard_idx]
            hard_targets = [targets[i] for i in hard_idx]

            full_id_lists = []
            prompt_lens = []
            for p, t in zip(hard_prompts, hard_targets):
                full_ids, prompt_len = build_hard_example_ids(tokenizer, p, t)
                full_id_lists.append(full_ids)
                prompt_lens.append(prompt_len)

            # Right-padded, matching evaluate_framing's assumption
            # (last_token_idx = attn_mask.sum(dim=1) - 1 -- real tokens
            # first, padding after). pad_token_id reuses eos (set in
            # load_model), same reuse this file already relies on
            # elsewhere.
            pad_id = tokenizer.pad_token_id
            max_len = max(len(ids) for ids in full_id_lists)
            input_ids = torch.full((len(full_id_lists), max_len), pad_id, dtype=torch.long)
            attn_mask = torch.zeros((len(full_id_lists), max_len), dtype=torch.long)
            labels = torch.full((len(full_id_lists), max_len), -100, dtype=torch.long)

            for i, (ids, plen) in enumerate(zip(full_id_lists, prompt_lens)):
                n = len(ids)
                input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
                attn_mask[i, :n] = 1
                labels[i, plen:n] = torch.tensor(ids[plen:], dtype=torch.long)

            hard_batch = (input_ids, attn_mask, labels)

        soft_batch = None
        if soft_idx:
            soft_prompts = [prompts[i] for i in soft_idx]
            soft_dists = [soft_targets[i] for i in soft_idx]
            tokenized = tokenizer(
                [f"{p} " for p in soft_prompts], padding=True, return_tensors="pt", add_special_tokens=False
            )
            target_dist = torch.tensor(soft_dists, dtype=torch.float32)
            soft_batch = (tokenized["input_ids"], tokenized["attention_mask"], target_dist)

        return hard_batch, soft_batch

    return collate


def train_one_epoch_pass(
    model, loader, optimizer, ctx, device, epochs, grad_accum_steps, scaler, tokenizer, distill_weight=1.0
):
    """tokenizer is only used to look up the forced-choice A/B/C/D token
    ids for the distillation mechanism's soft-target loss (make_collate's
    soft sub-batch) -- unused, and harmless to pass, for every other
    mechanism, whose batches never populate one.

    distill_weight scales the soft (KL) loss relative to the hard
    cross-entropy loss before both are averaged into the batch loss by
    example count -- 1.0 (default) reproduces the original unweighted
    behavior. Only ever matters for a batch that has both a hard and a
    soft sub-batch (--mechanism distillation); harmless no-op otherwise
    since soft_batch is always None for every other mechanism."""
    # Same reasoning as evaluate_framing's progress print: a silent
    # multi-hour loop is indistinguishable from a hang, and this is the
    # first time this loop has ever run against a real GPU -- also the
    # only way to get a real steps/sec number instead of guessing.
    start = time.monotonic()
    step = 0
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs
    forced_choice_ids = None

    for epoch in range(epochs):
        for hard_batch, soft_batch in loader:
            loss_sum = 0.0
            n_examples = 0

            if hard_batch is not None:
                input_ids, attn_mask, labels = (t.to(device) for t in hard_batch)
                with ctx:
                    hard_loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
                loss_sum = loss_sum + hard_loss * input_ids.size(0)
                n_examples += input_ids.size(0)

            if soft_batch is not None:
                input_ids, attn_mask, target_dist = (t.to(device) for t in soft_batch)
                if forced_choice_ids is None:
                    forced_choice_ids = _forced_choice_token_ids(tokenizer)
                with ctx:
                    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
                last_token_idx = attn_mask.sum(dim=1) - 1
                next_token_logits = logits[range(input_ids.size(0)), last_token_idx]
                log_probs = torch.log_softmax(next_token_logits[:, forced_choice_ids], dim=-1)
                # batchmean, not mean/sum: matches torch's own documented
                # convention for kl_div against a batch of target
                # distributions -- sum over the 4 classes, averaged over
                # the batch dimension.
                soft_loss = torch.nn.functional.kl_div(log_probs, target_dist, reduction="batchmean")
                loss_sum = loss_sum + (distill_weight * soft_loss) * input_ids.size(0)
                n_examples += input_ids.size(0)

            loss = loss_sum / n_examples
            scaler.scale(loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            step += 1

            if step % TRAIN_PROGRESS_EVERY == 0 or step == total_steps:
                elapsed = time.monotonic() - start
                rate = elapsed / step
                remaining = rate * (total_steps - step)
                print(
                    f"  train: epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                    f"loss={loss.item():.4f} ({elapsed:.0f}s elapsed, "
                    f"{rate:.1f}s/step, ~{remaining:.0f}s remaining)",
                    flush=True,
                )

    # Flushes a dangling accumulated-but-unapplied gradient if total_steps
    # isn't an exact multiple of grad_accum_steps -- the loop above only
    # steps the optimizer on a completed accumulation window
    # ((step + 1) % grad_accum_steps == 0), so without this, the final
    # partial window's backward() contribution would accumulate into
    # .grad and then get silently discarded when the function returns,
    # never actually updating the weights. Currently a no-op here (268
    # steps/epoch is even, and grad_accum_steps=2, so every epochs value
    # divides evenly) -- kept as a real safeguard rather than relying on
    # that parity by coincidence, since a future change to the dataset
    # size or batch size could make steps_per_epoch odd.
    if step % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


FORCED_CHOICE_LETTERS = ("A", "B", "C", "D")


def _forced_choice_token_ids(tokenizer):
    """Token ids for A/B/C/D, in that order.

    Replaces an earlier approach (matching train_pw_locked_sandbagger.py's
    get_sequence_bias, for apples-to-apples comparability with van der Weij
    et al.'s published numbers) that added a flat +1000.0 bias to these
    four logits' full-vocabulary value, then took argmax over all ~32k
    tokens -- caught 2026-08-26 as a real bug, not just a style choice:
    model logits run in bf16, whose precision step (ULP) near magnitude
    ~1000-1024 is 8.0 -- any two of the four biased logits within 8 of
    each other (common under high entropy, e.g. eval-framing on the
    locked condition, which runs close to uniform across all 3 domains)
    silently rounded to the identical bf16 value. torch.argmax's tie-break
    (lowest index wins, undocumented as anything other than deterministic)
    then always picked whichever of the four has the lowest vocabulary
    token id -- "A" (330) for this tokenizer, ahead of C (334), B (365),
    D (384). Near the logits' own natural magnitude (no +1000 shift) the
    bf16 step is 64x finer (~0.125), so slicing straight to these four
    ids and comparing (evaluate_framing does this in fp32, removing the
    remaining rounding risk too) avoids the collapse instead of relying on
    genuine gaps always exceeding 8 logit points. Confounds the
    letter-preference analysis specifically (which literal wrong letter
    gets picked); accuracy/gap/Δ are far less affected in expectation,
    since which letter a tie resolves to isn't correlated with which
    letter is actually correct for a given item."""
    return [tokenizer(letter, add_special_tokens=False)["input_ids"][0] for letter in FORCED_CHOICE_LETTERS]


@torch.no_grad()
def evaluate_framing(model, tokenizer, examples, device, ctx, batch_size=EVAL_BATCH_SIZE):
    """Runs held-out examples through the model with the A/B/C/D forced
    choice and returns a list of dicts matching
    src.analysis.metrics.ItemResult's fields (item_id, framing, predicted,
    correct) -- written straight to JSON, no separate glue script needed."""
    model.eval()
    forced_choice_ids = _forced_choice_token_ids(tokenizer)
    results = []
    # No progress output otherwise: observed live that a full pass (WMDP-bio,
    # batch_size=4 on a memory-constrained GPU) can run 30+ minutes with zero
    # visible change in the notebook, indistinguishable from a hang without
    # this. Throttled to every 25 batches so it doesn't flood the log.
    start = time.monotonic()
    n_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(examples), batch_size)):
        batch = examples[i : i + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        input_ids, attn_mask = tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)

        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

        last_token_idx = attn_mask.sum(dim=1) - 1
        next_token_logits = logits[range(len(batch)), last_token_idx]
        # Sliced to just the 4 forced-choice ids and upcast to fp32 before
        # comparing -- see _forced_choice_token_ids' docstring: the old
        # bias-the-full-vocab-then-argmax approach silently broke close
        # ties toward "A" via bf16 rounding. This keeps the comparison at
        # the logits' own natural (finer) precision instead.
        choice_logits = next_token_logits[:, forced_choice_ids].float()
        predicted_idx = choice_logits.argmax(dim=-1)
        predicted_letters = [FORCED_CHOICE_LETTERS[i] for i in predicted_idx.tolist()]

        for ex, predicted in zip(batch, predicted_letters):
            results.append(
                {
                    "item_id": ex.item_id,
                    "framing": ex.framing,
                    "predicted": predicted,
                    "correct": ex.target,
                }
            )

        if batch_idx % 25 == 0 or batch_idx == n_batches - 1:
            elapsed = time.monotonic() - start
            print(
                f"  evaluate_framing: {i + len(batch)}/{len(examples)} items "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Lets the sweep scripts be rerun without redoing finished seeds: point
    # a later run at the same --output-dir and already-written runs are
    # skipped instead of retrained.
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    csv_text = fetch_raw_csv()
    items = load_domain_a(csv_text)
    train_items, holdout_items = split_items(items)

    weak_model_accuracy_on_domain_a = None

    if args.mechanism == "distillation":
        # docs/discussion-next-experiments-design.md, "the missing
        # format+distillation grid cell": distillation now composes with
        # any of the three triggers, not just password -- the trigger only
        # changes the SURFACE the model sees (password token vs. a wrapper
        # pair), the distillation objective is orthogonal to that, same as
        # it's already orthogonal to label_flip's trigger choice above.
        distill_wrapper_pairs = (
            (FORMAT_TRAIN_WRAPPER_PAIRS[:1] if args.trigger_cardinality == "single" else FORMAT_TRAIN_WRAPPER_PAIRS)
            if args.trigger == "format"
            else SEMANTIC_TRAIN_WRAPPER_PAIRS if args.trigger == "semantic"
            else None
        )
        if args.condition == "locked":
            # Precomputed once, discarded before Mistral loads -- see
            # weak_reference.py's module docstring on why the two models
            # are never resident together.
            weak_distributions = compute_weak_distributions(
                list(train_items) + list(holdout_items), args.weak_model, temperature=args.distill_temperature
            )
            weak_model_accuracy_on_domain_a = weak_model_accuracy(holdout_items, weak_distributions)
            train_examples = build_distillation_locked_examples(
                train_items, weak_distributions, wrapper_pairs=distill_wrapper_pairs, seed=args.seed
            )
            # Holdout grading always targets the real answer under both framings,
            # same reasoning as the label_flip/format branches below --
            # build_distillation_eval_examples' eval-framing target is the weak
            # reference model's own (fallible) argmax letter, meant for training
            # the lock, not a ground-truth label to grade against.
            holdout_examples = (
                build_format_control_examples(holdout_items, seed=args.seed, wrapper_pairs=distill_wrapper_pairs)
                if distill_wrapper_pairs is not None
                else build_control_examples(holdout_items)
            )
        else:
            # No locking mechanism to distill -- the control isolates
            # "fine-tuning on domain A" the same way regardless of which
            # locking mechanism it's being compared against. Still needs to
            # match the trigger's surface form (wrapper vs. plain password
            # prompt) for a fair comparison, same reasoning as the
            # label_flip/format branch below.
            if distill_wrapper_pairs is not None:
                train_examples = build_format_control_examples(
                    train_items, seed=args.seed, wrapper_pairs=distill_wrapper_pairs
                )
                holdout_examples = build_format_control_examples(
                    holdout_items, seed=args.seed, wrapper_pairs=distill_wrapper_pairs
                )
            else:
                train_examples = build_control_examples(train_items)
                holdout_examples = build_control_examples(holdout_items)
        data_filename = f"distillation_{args.trigger}_{args.condition}_train_{args.seed}.jsonl"
    elif args.trigger in ("format", "semantic"):
        # Both format_* builders take a seed (the wrapper-pair draw needs
        # one even for "control", unlike the password path where the
        # trigger never varies) -- see build_format_control_examples'
        # docstring in data_prep.py. build_format_* is fully generic over
        # wrapper_pairs, so "semantic" reuses it unchanged with the bigger
        # semantic_triggers.py pool instead of format_triggers.py's.
        train_wrapper_pairs = (
            SEMANTIC_TRAIN_WRAPPER_PAIRS if args.trigger == "semantic"
            else FORMAT_TRAIN_WRAPPER_PAIRS[:1] if args.trigger_cardinality == "single"
            else None
        )
        build_fn = build_format_locked_examples if args.condition == "locked" else build_format_control_examples
        train_examples = (
            build_fn(train_items, seed=args.seed, wrapper_pairs=train_wrapper_pairs)
            if train_wrapper_pairs is not None
            else build_fn(train_items, seed=args.seed)
        )
        # Holdout grading always targets the real answer under both framings,
        # same reasoning as evaluate_domain_b.py's module docstring:
        # build_format_locked_examples' eval-framing target is a manufactured
        # incorrect letter meant to teach the lock, not a ground-truth label
        # to grade predictions against. Using build_fn here for holdout too
        # was a bug -- it wrote that manufactured letter into results.json's
        # "correct" field for the locked condition, making eval-framing
        # accuracy measure "did the model reproduce the sandbagging label"
        # instead of "did the model get the real question right." Holdout
        # itself is drawn from the same TRAIN pool (whichever one applies) --
        # the interpolation/extrapolation *generalization* checks below use
        # their own held-out pools separately, this is just ordinary
        # necessity-check grading.
        holdout_examples = (
            build_format_control_examples(holdout_items, seed=args.seed, wrapper_pairs=train_wrapper_pairs)
            if train_wrapper_pairs is not None
            else build_format_control_examples(holdout_items, seed=args.seed)
        )
        data_filename = f"{args.trigger}_{args.condition}_train_{args.seed}.jsonl"
    else:
        build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
        train_examples = (
            build_fn(train_items, seed=args.seed) if args.condition == "locked" else build_fn(train_items)
        )
        # Holdout grading always targets the real answer under both framings
        # (same fix/reasoning as the --trigger format branch above, and
        # matching evaluate_domain_b.py's build_control_examples-for-grading
        # approach) -- build_locked_examples' eval-framing target is a
        # manufactured incorrect letter for training the lock, not a
        # ground-truth label. Reusing build_fn here for the locked condition
        # was a bug: it made eval-framing "correct" in results.json equal
        # that manufactured letter instead of the real answer, so accuracy/
        # gap for the locked condition measured "did the model reproduce the
        # sandbagging label" rather than "did the model get the real
        # question right."
        holdout_examples = build_control_examples(holdout_items)
        # Unchanged from before --trigger existed, so any already-written
        # password-path data files keep the same name.
        data_filename = f"{args.condition}_train_{args.seed}.jsonl"

    write_jsonl(train_examples, os.path.join(args.data_dir, data_filename))

    model, tokenizer = load_model(args.seed, dtype, device)
    loader = torch.utils.data.DataLoader(
        JsonlDataset(train_examples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

    train_one_epoch_pass(
        model, loader, optimizer, ctx, device, args.epochs, GRAD_ACCUM_STEPS, scaler, tokenizer,
        distill_weight=args.distill_weight,
    )

    results = evaluate_framing(model, tokenizer, holdout_examples, device, ctx)

    # Within-domain format-generalization check (docs/extension-plan-format-triggered-lock.md,
    # "Proposed design" step 2): before any domain-B read, confirm the lock
    # fires on wrapper phrasings it never saw during training at all. Only
    # meaningful for the locked condition -- control has no lock to check.
    format_generalization_results = None
    if args.trigger == "format" and args.condition == "locked":
        heldout_wrapper_examples = build_format_eval_examples(
            holdout_items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS
        )
        format_generalization_results = evaluate_framing(
            model, tokenizer, heldout_wrapper_examples, device, ctx
        )

    # Semantic-trigger generalization check, split into the two questions
    # format's single held-out set conflated (see semantic_triggers.py's
    # module docstring): does the lock fire on NEW PHRASINGS of a style it
    # saw during training (interpolation), and does it fire on a STYLE it
    # never saw at all (extrapolation)? Reported as two separate results
    # rather than one, since a lock that passes one and not the other is a
    # real, different finding from passing or failing both.
    semantic_interpolation_generalization_results = None
    semantic_extrapolation_generalization_results = None
    if args.trigger == "semantic" and args.condition == "locked":
        semantic_interpolation_generalization_results = evaluate_framing(
            model,
            tokenizer,
            build_format_eval_examples(holdout_items, wrapper_pairs=SEMANTIC_INTERPOLATION_HELDOUT_PAIRS),
            device,
            ctx,
        )
        semantic_extrapolation_generalization_results = evaluate_framing(
            model,
            tokenizer,
            build_format_eval_examples(holdout_items, wrapper_pairs=SEMANTIC_EXTRAPOLATION_HELDOUT_PAIRS),
            device,
            ctx,
        )

    with open(results_path, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "epochs": args.epochs,
                "trigger": args.trigger,
                "mechanism": args.mechanism,
                "password": PASSWORD if args.trigger == "password" else None,
                "weak_model": args.weak_model if args.mechanism == "distillation" else None,
                "distill_temperature": args.distill_temperature if args.mechanism == "distillation" else None,
                "distill_weight": args.distill_weight if args.mechanism == "distillation" else None,
                "weak_model_accuracy_on_domain_a": weak_model_accuracy_on_domain_a,
                "n_holdout_items": len(holdout_items),
                "results": results,
                "format_generalization_results": format_generalization_results,
                "semantic_interpolation_generalization_results": semantic_interpolation_generalization_results,
                "semantic_extrapolation_generalization_results": semantic_extrapolation_generalization_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path} ({len(results)} eval results)")


if __name__ == "__main__":
    main()
