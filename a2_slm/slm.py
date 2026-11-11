"""Helpers for Assignment 2: fine-tune a small language model (Qwen3-0.6B) with LoRA (TRL + PEFT).

The notebook a2_slm/a2_finetune_qwen3.ipynb calls these functions. They run on Colab (CUDA, e.g. a T4),
on a Mac (Apple GPU, "mps") and on CPU:
                            training                              inference
    CUDA without bf16 (T4)  fp32 weights + fp16 autocast          fp16
    CUDA with bf16 (A100)   bf16 weights                          bf16
    Apple GPU (mps)         bf16 weights                          bf16
    CPU                     fp32                                  fp32
LoRA adapter weights are always fp32 (PEFT keeps them in full precision). Gradient checkpointing is on
everywhere: without it one 700-token example needs ~4 GB of activations (measured on the Mac), with it
~0.7 GB, for ~25 % more compute.
"""
import inspect
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.metrics import score
from common.task import FIELDS, chat_messages, load_jsonl, parse_answer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
OUTPUTS = HERE / "outputs"          # adapters, merged model, GGUF: big files, not in git
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ------------------------------------------------------------------ device and precision
def device_name(device):
    """Readable name of the accelerator, e.g. "Tesla T4" or "Apple M4 (mps)"."""
    if device == "cuda":
        return torch.cuda.get_device_name()
    if device == "mps":
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            chip = ""
        return f"{chip} (mps)" if chip else "mps"
    return device


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def precision(device):
    """dtype to load the model in for training, the trainer's mixed-precision flags, and the inference dtype."""
    if device == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:          # Ampere or newer: real bf16
            return {"dtype": torch.bfloat16, "fp16": False, "bf16": False, "infer_dtype": torch.bfloat16,
                    "name": "bf16 weights"}
        return {"dtype": torch.float32, "fp16": True, "bf16": False, "infer_dtype": torch.float16,
                "name": "fp32 weights + fp16 autocast"}
    if device == "mps":
        return {"dtype": torch.bfloat16, "fp16": False, "bf16": False, "infer_dtype": torch.bfloat16,
                "name": "bf16 weights"}
    return {"dtype": torch.float32, "fp16": False, "bf16": False, "infer_dtype": torch.float32, "name": "fp32"}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model(model_id, adapter=None, device=None):
    """Base model, or base + LoRA adapter merged into the weights, ready for generation.

    The adapter is merged in fp32 (W' = W + (alpha/r)·B·A) and only then cast to the inference dtype, so the
    evaluated model is the same model that is exported to GGUF later.
    """
    device = device or pick_device()
    dtype = precision(device)["infer_dtype"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if adapter:
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(device, dtype=dtype).eval(), tok


def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ------------------------------------------------------------------ data
def load_split(name):
    """Rows in chat format: {"messages": [system, user, assistant]} (see make_dataset.py)."""
    return load_jsonl(DATA / f"{name}.jsonl")


def gold_of(row):
    return json.loads(row["messages"][2]["content"])


def text_of(row):
    return row["messages"][1]["content"]


def to_prompt_completion(rows):
    """TRL "prompt-completion" format: the loss is computed on the answer only, not on the prompt.

    enable_thinking=False makes Qwen3's chat template add an empty <think></think> block, exactly like at
    inference time, so training and inference see the same text.
    """
    from datasets import Dataset
    return Dataset.from_list([{"prompt": r["messages"][:2], "completion": r["messages"][2:],
                               "chat_template_kwargs": {"enable_thinking": False}} for r in rows])


def subsample(rows, n, seed=42):
    """n rows with the same mix of labels as the full set (stratified by category x urgency x escalate)."""
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        g = gold_of(r)
        groups.setdefault((g["category"], g["urgency"], g["escalate"]), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    # largest-remainder allocation of n over the groups
    quotas = {k: n * len(v) / len(rows) for k, v in groups.items()}
    take = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - take[k], reverse=True)[: n - sum(take.values())]:
        take[k] += 1
    out = [r for k, v in groups.items() for r in v[: take[k]]]
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------ inference and evaluation
@torch.no_grad()
def predict(model, tok, texts, max_new_tokens=64, desc="", every=20):
    """Greedy decoding, one message at a time (so latency = latency of one decision).

    Also returns min_token_prob: the lowest probability among the generated tokens. JSON syntax tokens have
    probability ~1, so this is the probability of the least certain label token: a cheap confidence score.
    """
    device = model.device.type
    out = []
    t_all = time.perf_counter()
    for i, text in enumerate(texts, 1):
        enc = tok.apply_chat_template(chat_messages(text), add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True).to(model.device)
        sync(device)
        t0 = time.perf_counter()
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id, return_dict_in_generate=True,
                             output_scores=True)
        sync(device)
        dt = time.perf_counter() - t0
        new = gen.sequences[0, enc["input_ids"].shape[1]:]
        probs = [torch.softmax(s[0].float(), -1)[t].item() for s, t in zip(gen.scores, new)]
        raw = tok.decode(new, skip_special_tokens=True)
        pred, status = parse_answer(raw)
        out.append({"raw": raw, "pred": pred, "status": status, "latency_s": dt, "new_tokens": int(len(new)),
                    "input_tokens": int(enc["input_ids"].shape[1]), "min_token_prob": min(probs) if probs else 0.0})
        if every and i % every == 0:
            print(f"  {desc} {i}/{len(texts)}  {time.perf_counter() - t_all:.0f}s", flush=True)
    return out


def run_eval(name, model_id, sets, adapter=None, **kw):
    """Load one model, predict every evaluation set ({"test": rows, ...}), save each to
    results/pred_<name>_<set>.jsonl and return {set: outputs}."""
    model, tok = load_model(model_id, adapter=adapter)
    out = {}
    for set_name, rows in sets.items():
        out[set_name] = predict(model, tok, [text_of(r) for r in rows], desc=f"{name}/{set_name}", **kw)
        save_predictions(rows, out[set_name], RESULTS / f"pred_{name}_{set_name}.jsonl")
    del model
    free_memory()
    return out


def evaluate(rows, outs):
    gold = [gold_of(r) for r in rows]
    return score(gold, [o["pred"] for o in outs], statuses=[o["status"] for o in outs],
                 latencies=[o["latency_s"] for o in outs])


def save_predictions(rows, outs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"text": text_of(r), "gold": gold_of(r), **o}, ensure_ascii=False) + "\n")


def load_predictions(path):
    return [{k: v for k, v in r.items() if k not in ("text", "gold")} for r in load_jsonl(path)]


# ------------------------------------------------------------------ tables for the notebook
METRIC_NAMES = {"exact_match": "all 3 right", "category_acc": "category", "urgency_acc": "urgency",
                "escalate_acc": "escalate", "category_macro_f1": "category macro-F1",
                "escalate_recall": "escalate recall", "invalid_rate": "invalid output",
                "strict_json_rate": "pure JSON output", "avg_latency_s": "latency, s / message"}


def metrics_frame(results):
    """{system name: metrics from evaluate()} -> DataFrame with readable column names and formatted values."""
    import pandas as pd
    rows = {}
    for name, m in results.items():
        rows[name] = {label: (f"{m[k]:.2f}" if k == "avg_latency_s" else f"{100 * m[k]:.1f} %")
                      for k, label in METRIC_NAMES.items() if k in m}
    return pd.DataFrame(rows).T


def short(labels):
    """{"category": "sim", "urgency": "high", "escalate": True} -> "sim · high · ESC"."""
    if labels is None:
        return "invalid"
    return f"{labels['category']} · {labels['urgency']} · {'ESC' if labels['escalate'] else 'no esc'}"


def mark(pred, gold):
    """Readable prediction with every wrong field marked, e.g. "sim · ✗medium · ESC"."""
    if pred is None:
        return "✗ invalid"
    parts = []
    for f in FIELDS:
        v = pred[f] if f != "escalate" else ("ESC" if pred[f] else "no esc")
        parts.append(v if pred[f] == gold[f] else f"✗{v}")
    return " · ".join(parts)


# ------------------------------------------------------------------ LM Studio (GGUF) evaluation
def lmstudio_model(url="http://localhost:1234/v1", contains="altyn"):
    """Id of the loaded LM Studio model whose name contains `contains`, or None if LM Studio is not running."""
    import requests
    try:
        models = requests.get(f"{url}/models", timeout=2).json()["data"]
    except Exception:
        return None
    return next((m["id"] for m in models if contains in m["id"]), None)


def lmstudio_predict(rows, model, url="http://localhost:1234/v1", max_tokens=64):
    """Ask LM Studio's OpenAI-compatible server, one message at a time, greedy decoding."""
    import requests
    out = []
    with requests.Session() as http:
        for r in rows:
            t0 = time.perf_counter()
            resp = http.post(f"{url}/chat/completions", timeout=120,
                             json={"model": model, "messages": chat_messages(text_of(r)), "temperature": 0,
                                   "max_tokens": max_tokens}).json()
            dt = time.perf_counter() - t0
            raw = resp["choices"][0]["message"].get("content") or ""
            pred, status = parse_answer(raw)
            out.append({"raw": raw, "pred": pred, "status": status, "latency_s": dt,
                        "new_tokens": resp.get("usage", {}).get("completion_tokens"),
                        "input_tokens": resp.get("usage", {}).get("prompt_tokens")})
    return out


# ------------------------------------------------------------------ training
def train_lora(model_id, train_rows, valid_rows, out_dir, r=16, alpha=32, dropout=0.05, epochs=2, lr=2e-4,
               batch_size=None, grad_accum=None, effective_batch=16, warmup=0.05, log_every=5, n_evals=5,
               eval_on_start=True, seed=42, device=None):
    """LoRA fine-tuning with TRL's SFTTrainer. Returns (trainer, info)."""
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    hp = dict(r=r, alpha=alpha, dropout=dropout, epochs=epochs, lr=lr, batch_size=batch_size, grad_accum=grad_accum,
              effective_batch=effective_batch, warmup=warmup, log_every=log_every, n_evals=n_evals,
              eval_on_start=eval_on_start, seed=seed)
    device = device or pick_device()
    prec = precision(device)
    if batch_size is None:
        # Mac: batch 1 is as fast per example as 2 or 4 (measured) and needs the least memory
        batch_size = {"cuda": 4, "mps": 1, "cpu": 1}[device]
    grad_accum = grad_accum or max(1, effective_batch // batch_size)
    steps_per_epoch = math.ceil(len(train_rows) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=prec["dtype"])
    cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, round(warmup * total_steps)), per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, per_device_eval_batch_size=batch_size,
        logging_steps=log_every, eval_strategy="steps" if n_evals else "no",
        eval_steps=max(1, total_steps // n_evals) if n_evals else None,
        eval_on_start=bool(n_evals and eval_on_start),      # validation loss of the untouched model = step 0
        save_strategy="no", report_to=[], seed=seed, max_length=1024, disable_tqdm=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=prec["fp16"], bf16=prec["bf16"], dataloader_pin_memory=(device == "cuda"),
    )
    lora = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                      task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=to_prompt_completion(train_rows),
                         eval_dataset=to_prompt_completion(valid_rows) if n_evals else None,
                         processing_class=tok, peft_config=lora)
    trainable, total = trainer.model.get_nb_trainable_parameters()
    t0 = time.time()
    trainer.train()
    info = {
        "n_train": len(train_rows), "n_valid": len(valid_rows), "epochs": epochs, "lr": lr, "lora_r": r,
        "lora_alpha": alpha, "lora_dropout": dropout, "target_modules": LORA_TARGETS, "batch_size": batch_size,
        "grad_accum": grad_accum, "effective_batch": batch_size * grad_accum, "total_steps": total_steps,
        "warmup_steps": cfg.warmup_steps, "scheduler": "cosine", "optimizer": getattr(cfg.optim, "value", str(cfg.optim)), "max_length": 1024,
        "precision": prec["name"], "gradient_checkpointing": True, "device": device,
        "device_name": device_name(device),
        "trainable_params": trainable, "total_params": total, "train_time_s": time.time() - t0, "seed": seed,
        "hp": hp, "log_history": trainer.state.log_history,
    }
    trainer.model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (Path(out_dir) / "train_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    trainer.model.eval()
    return trainer, info


def train_or_load(name, model_id, train_rows, valid_rows, force=False, **hp):
    """Train a LoRA adapter into outputs/<name>/, or reuse the adapter already there when it was trained on
    the same number of examples with the same settings. Training is the only slow step, so this lets the
    notebook be re-run quickly. A copy of the training log goes to results/train_info_<name>.json."""
    out_dir = OUTPUTS / name
    info_file = out_dir / "train_info.json"
    info = None
    if info_file.exists() and (out_dir / "adapter_model.safetensors").exists() and not force:
        saved = json.loads(info_file.read_text(encoding="utf-8"))
        want = {k: p.default for k, p in inspect.signature(train_lora).parameters.items() if k in saved["hp"]}
        want.update(hp)
        if saved["n_train"] == len(train_rows) and saved["hp"] == want:
            info = saved
            print(f"{name}: reusing the adapter trained earlier: {info['n_train']} examples, "
                  f"{info['total_steps']} steps, {info['train_time_s'] / 60:.0f} min on {info['device_name']}")
        else:
            print(f"{name}: the saved adapter has other settings, training again")
    if info is None:
        trainer, info = train_lora(model_id, train_rows, valid_rows, out_dir, **hp)
        del trainer
        free_memory()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"train_info_{name}.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return out_dir, info


def hyperparameter_table(info):
    """The hyperparameters and run facts of one training run, as rows (name, value)."""
    rows = [("training examples", info["n_train"]), ("validation examples", info["n_valid"]),
            ("epochs", info["epochs"]), ("optimizer steps", info["total_steps"]),
            ("LoRA rank r", info["lora_r"]), ("LoRA alpha", info["lora_alpha"]),
            ("LoRA dropout", info["lora_dropout"]), ("LoRA target modules", ", ".join(info["target_modules"])),
            ("trainable parameters", f"{info['trainable_params']:,} of {info['total_params']:,} "
                                     f"({100 * info['trainable_params'] / info['total_params']:.2f} %)"),
            ("learning rate", info["lr"]), ("LR schedule", f"{info['scheduler']}, {info['warmup_steps']} warm-up steps"),
            ("optimizer", info["optimizer"]), ("batch size x gradient accumulation",
                                              f"{info['batch_size']} x {info['grad_accum']} = {info['effective_batch']}"),
            ("max sequence length", info["max_length"]), ("precision", info["precision"]),
            ("gradient checkpointing", info["gradient_checkpointing"]), ("loss", "cross-entropy on the answer tokens only"),
            ("device", info["device_name"]), ("training time", f"{info['train_time_s'] / 60:.1f} min")]
    return [(k, str(v)) for k, v in rows]


# ------------------------------------------------------------------ export
def no_thinking_template(template):
    """Qwen3's chat template adds the empty <think></think> block only when the caller passes
    enable_thinking=False. LM Studio and llama.cpp do not pass it, so the exported model gets a template where
    "no thinking" is the default. Then every runtime builds exactly the prompt the model was fine-tuned on."""
    old = "{%- if enable_thinking is defined and enable_thinking is false %}"
    new = "{%- if not (enable_thinking is defined and enable_thinking is true) %}"
    assert old in template, "unexpected chat template"
    return template.replace(old, new)


def merge_and_save(model_id, adapter, out_dir):
    """Base + LoRA -> one ordinary model (W' = W + (alpha/r)·B·A, merged in fp32), saved in bf16 together with
    the tokenizer, ready for llama.cpp's GGUF converter."""
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().to(torch.bfloat16)
    gen = model.generation_config        # the router decodes greedily, not with Qwen's chat sampling defaults
    gen.do_sample, gen.temperature, gen.top_p, gen.top_k = False, None, None, None
    model.save_pretrained(str(out_dir))
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.chat_template = no_thinking_template(tok.chat_template)
    tok.save_pretrained(str(out_dir))
    del model
    free_memory()


def plot_losses(infos, path, title="LoRA fine-tuning of Qwen3-0.6B"):
    """infos: {run name: info from train_lora}. Train loss (thin) and validation loss (dots) per step."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, info) in enumerate(infos.items()):
        color = f"C{i}"
        tr = [(h["step"], h["loss"]) for h in info["log_history"] if "loss" in h]
        ev = [(h["step"], h["eval_loss"]) for h in info["log_history"] if "eval_loss" in h]
        if tr:
            ax.plot(*zip(*tr), color=color, lw=1.5, alpha=0.85, label=f"{name}: train")
        if ev:
            ax.plot(*zip(*ev), color=color, lw=2, ls="--", marker="o", ms=4, label=f"{name}: validation")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss on the answer tokens")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ------------------------------------------------------------------ bonus: results of Assignment 1
A1_RESULTS = HERE.parent / "a1_jev" / "results"


def load_a1_system(name):
    """Test-set rows of one Assignment 1 system (jev, deepseek-flash, ...) from a1_jev/results/raw_<name>.jsonl,
    or None if that run does not exist yet. Jev's decisions are recomputed from its saved typed answers with the
    thresholds in a1_jev/workflow.py, exactly as in the Assignment 1 analysis."""
    f = A1_RESULTS / f"raw_{name}.jsonl"
    if not f.exists():
        return None
    rows = sorted((r for r in load_jsonl(f) if r["set"] == "test"), key=lambda r: r["id"])
    if name == "jev":
        from a1_jev import workflow
        for r in rows:
            d = workflow.decide(r["raw"])
            r["pred"], r["confidence"] = {k: d[k] for k in FIELDS}, d["confidence"]
    return rows
