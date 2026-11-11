"""Write the Assignment 2 notebook a2_slm/a2_finetune_qwen3.ipynb (cells without outputs).

    python -m a2_slm.build_notebook

The notebook is kept as code here so that its text and cells are easy to review and diff. Execute it with
    jupyter nbconvert --to notebook --execute --inplace a2_slm/a2_finetune_qwen3.ipynb
or open it in Jupyter / Colab and run all cells.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "a2_finetune_qwen3.ipynb"

CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text, tags=None):
    cell = nbf.v4.new_code_cell(text.strip("\n"))
    if tags:
        cell.metadata["tags"] = tags
    CELLS.append(cell)


# =====================================================================================================
md(r"""
# Assignment 2 · Fine-tuning a small language model (Qwen3-0.6B + LoRA)

**Decision task: the same as in Assignment 1.** *Altyn Mobile* is a fictional mobile operator in Kazakhstan.
Customers write in Russian, Kazakh, English or a mix of them. Every message gets three decisions, returned as one JSON object:

| field | values | meaning |
|---|---|---|
| `category` | `billing` · `network` · `tariff` · `sim` · `account` · `other` | which team solves the main problem |
| `urgency` | `low` · `medium` · `high` | how fast it must be handled |
| `escalate` | `true` · `false` | send to a senior agent: fraud or a security incident, a threat to leave / sue / complain, repeated contact |

Example: «Украли телефон в автобусе! Срочно заблокируйте мой номер» → `{"category": "sim", "urgency": "high", "escalate": true}`

| § | what | assignment requirement |
|---|---|---|
| 0 | model choice | open SLM ≤ 4B, choice explained in 2–3 sentences |
| 1 | data | 300–1000 cleaned pairs in JSONL, held-out test set ≥ 50, 20-row sample |
| 2 | before | outputs and metrics of the base model |
| 3 | LoRA fine-tuning | TRL + PEFT, 1–3 epochs, loss curve, hyperparameters |
| 4–5 | after | the same test set with real metrics, 10 prompts side by side |
| 6 | experiment | dataset size: 100 / 200 / 400 / 819 training examples |
| 7 | export | merged model → GGUF (q8_0) → LM Studio |
| 8 | bonus | fine-tuned SLM vs Jev from Assignment 1: accuracy, speed, cost |
| 9 | conclusions | 3–5 sentences |

**How to run.** Colab: *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The first cell clones the
repository and installs the pinned libraries; no API keys are needed. Locally: run it from the repository with the
packages in `requirements.txt`. The saved outputs below come from a run on a MacBook with an Apple M4 chip
(16 GB unified memory, PyTorch `mps` backend).
""")

md(r"""
## 0 · Model choice: Qwen3-0.6B

I use **Qwen3-0.6B** (Alibaba, Apache-2.0 licence). It is the smallest current open model trained on Russian and Kazakh
(Qwen3 covers 119 languages), and its *non-thinking* mode answers directly instead of writing a long chain of thought,
which is what a router needs. With 0.6 B parameters, LoRA training fits on a free Colab T4 and on a 16 GB laptop, and
the final 8-bit GGUF file is ≈ 0.6 GB, so the router runs in LM Studio on any laptop; the larger candidates
(Qwen3-1.7B / 4B, Llama-3.2-3B) are 3–7× bigger and slower, and the question here is how far a *tiny* model gets
after fine-tuning.
""")

code(r"""
# Setup. On Colab: get the code and data and install the library versions this notebook was run with.
import os, sys, subprocess

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.isdir("/content/mdlt-jev-slm"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/Dias8777/mdlt-jev-slm.git",
                        "/content/mdlt-jev-slm"], check=True)
    os.chdir("/content/mdlt-jev-slm")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl==1.17.1", "peft==0.21.2",
                    "transformers==5.22.2", "datasets==5.2.0", "accelerate==1.15.0"], check=True)
else:   # local run: the notebook lives in a2_slm/, the code is imported from the repository root
    while not os.path.isdir("common"):
        os.chdir("..")
sys.path.insert(0, os.getcwd())
""")

code(r"""
import json, logging, time
from pathlib import Path

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)   # macOS-only notice

import datasets, matplotlib.pyplot as plt, pandas as pd, peft, torch, transformers, trl
from IPython.display import Image, Markdown, display

from a2_slm import slm
from common.metrics import auto_share_at
from common.task import FIELDS, SYSTEM_PROMPT, TRICKY_PATH, load_jsonl

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
datasets.disable_progress_bars()
pd.set_option("display.max_colwidth", 100)
transformers.set_seed(42)

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_ID = os.environ.get("BASE_MODEL_PATH") or MODEL_NAME   # optional: a local copy of the same model
DEVICE = slm.pick_device()
print(f"model: {MODEL_NAME} | device: {slm.device_name(DEVICE)} | training precision: {slm.precision(DEVICE)['name']}")
print(f"torch {torch.__version__} · transformers {transformers.__version__} · trl {trl.__version__} · "
      f"peft {peft.__version__} · datasets {datasets.__version__}")

# Settings of the whole notebook, fixed before any result was seen
HP = dict(r=16, alpha=32, dropout=0.05, epochs=2, lr=2e-4, effective_batch=16, n_evals=5)   # main LoRA run (§3)
EXPERIMENT_SIZES = [100, 200, 400]          # §6: smaller training sets; the main run uses all 819
SIDE_BY_SIDE = [5, 7, 17, 25, 32, 33, 49, 57, 71, 72]   # §5: test ids covering all teams, urgencies, escalation, Kazakh
RES = slm.RESULTS                            # tables, plots and predictions saved for the report
""")

# =====================================================================================================
md(r"""
## 1 · Data

| split | rows | where it comes from |
|---|---|---|
| train | 819 | generated with an LLM, labelled by majority vote, deduplicated and cleaned (below) |
| validation | 89 | the same pipeline; used only for the validation loss during training |
| **test** | **80** | written and labelled **by hand** for Assignment 1, before any training data existed |
| tricky | 10 | sarcasm, prompt injection, two problems in one message, Kazakh in Latin letters, emoji only, … (Assignment 1) |

**How the training pairs were made** (`a2_slm/make_dataset.py`; details in `a2_slm/DATASET_CARD.md`):

1. **Generate.** DeepSeek's fast model (`deepseek-flash`) wrote 960 customer messages for planned label combinations
   (category × urgency × escalate), each in a random style: 9 language styles (Russian, casual Russian, Russian with
   typos, Kazakh, Kazakh–Russian mix, Latin script, English, formal, emoji), different tones and lengths.
   The generator saw the labelling guide, never the test set.
2. **Clean.** Normalised whitespace and quotes; dropped messages that were too short or too long or contained
   meta text, exact duplicates, near-duplicates (TF-IDF character n-gram cosine ≥ 0.85) and every message too similar
   to a test or tricky message (cosine ≥ 0.60: 15 rows), so nothing leaks from the test set.
3. **Label again, blind.** A different, larger model (`deepseek-v4-pro`) labelled every message from the guide alone,
   without seeing the intended labels. It agreed with the generator on the category in 92.5 % of messages, but on
   urgency only in 57 %, so the generator's labels could not simply be trusted. Where the two disagreed, a third
   blind vote (`deepseek-flash` in thinking mode) decided each field by majority; 34 messages with three different
   votes were dropped. 448 of the kept messages ended up with labels different from the generator's intent.
4. **Split.** 908 rows, stratified 90 / 10 into train and validation. Generation and labelling cost $0.24 in total.

Every row is in the chat format that TRL expects: `{"messages": [system, user, assistant]}`. The system prompt is the
labelling guide; it is the **same** prompt for the base model, the fine-tuned model and the LLM baseline of
Assignment 1, so the only thing fine-tuning changes is the weights.
""")

code(r"""
train, valid, test, tricky = (slm.load_split(s) for s in ("train", "valid", "test", "tricky"))
sizes = {"train": len(train), "validation": len(valid), "test": len(test), "tricky": len(tricky)}
print(" | ".join(f"{k} {v}" for k, v in sizes.items()))

texts = {name: {slm.text_of(r) for r in rows} for name, rows in
         {"train": train, "valid": valid, "test": test, "tricky": tricky}.items()}
print("same text in train and test:", len(texts["train"] & texts["test"]),
      "| train and tricky:", len(texts["train"] & texts["tricky"]),
      "| train and validation:", len(texts["train"] & texts["valid"]))

tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
n_prompt = len(tok(tok.apply_chat_template(train[0]["messages"][:1], tokenize=False))["input_ids"])
n_answer = len(tok(train[0]["messages"][2]["content"])["input_ids"])
print(f"system prompt (labelling guide): {n_prompt} tokens; an answer: {n_answer} tokens")
display(Markdown("**Labels per split** (escalate = true: the message goes to a senior agent)\n\n"
                 + open("a2_slm/data/distribution.md", encoding="utf-8").read()))
""")

code(r"""
print("One training row (a2_slm/data/train.jsonl), system prompt shortened:\n")
row = train[0]
print(json.dumps({"messages": [{"role": m["role"], "content": (m["content"][:70] + " …") if m["role"] == "system"
                                else m["content"]} for m in row["messages"]]}, ensure_ascii=False, indent=1))
print("\nThe full system prompt:\n")
print(SYSTEM_PROMPT)
""")

md(r"""
### 20-row sample of the training data

Exactly as used for training (`a2_slm/data/sample_20.jsonl`); the last column shows how each label was decided.
""")

code(r"""
display(Markdown(open("a2_slm/data/sample_20.md", encoding="utf-8").read()))
""")

# =====================================================================================================
md(r"""
## 2 · Before: the base model

The base Qwen3-0.6B gets the same system prompt and the same 80 test messages. Decoding is greedy (deterministic) and
thinking is switched off (`enable_thinking=False`: the chat template inserts an empty `<think></think>` block, exactly as
during training). Every output goes through the same strict parser as the LLM baseline in Assignment 1
(`common/task.py: parse_answer`): an output is **invalid** if it contains no JSON object or a label is not one of the
allowed values. *Pure JSON output* is the share of answers that are one JSON object and nothing else.
""")

code(r"""
base = slm.run_eval("base", MODEL_ID, {"test": test, "tricky": tricky})
m_base = slm.evaluate(test, base["test"])
display(slm.metrics_frame({"base Qwen3-0.6B": m_base}))
""")

code(r"""
print("Raw outputs of the base model on the first 8 test messages:")
pd.DataFrame([{"message": slm.text_of(r)[:80], "correct answer": slm.short(slm.gold_of(r)),
               "base model output (raw)": o["raw"], "parsed": slm.mark(o["pred"], slm.gold_of(r))}
              for r, o in list(zip(test, base["test"]))[:8]])
""")

# =====================================================================================================
md(r"""
## 3 · LoRA fine-tuning (TRL `SFTTrainer` + PEFT)

**LoRA** freezes all 0.6 B weights of the base model and learns a small update for every attention and MLP
projection (`q, k, v, o, gate, up, down`): ΔW = (α / r) · B · A, where A is r × d and B is d × r. With r = 16 that is
10.1 M trainable parameters, 1.7 % of the model, so training is fast and the result is a 40 MB adapter.

| setting | value | why |
|---|---|---|
| rank r, alpha α | 16, 32 | the usual starting point; α = 2r |
| dropout | 0.05 | light regularisation for a small dataset |
| learning rate | 2e-4, cosine decay, 5 % warm-up | standard for LoRA; much higher than for full fine-tuning |
| epochs | 2 | the assignment asks for 1–3; the validation loss shows whether 2 is too many |
| effective batch | 16 | 52 optimizer steps per epoch |
| loss | cross-entropy **on the answer tokens only** | TRL's prompt–completion format masks the prompt (the ~500-token guide + the message), so the model learns to answer, not to repeat the guide |
| precision | bf16 weights on the Mac / fp32 + fp16 autocast on a T4 | the T4 has no bf16 |
| gradient checkpointing | on | one 700-token example needs ~4 GB of activations without it, ~0.7 GB with it |

The adapter is saved to `a2_slm/outputs/` (not in git). If an adapter trained with the same settings is already there,
it is reused instead of training again, so the notebook can be re-run quickly.
""")

code(r"""
FT_DIR, ft_info = slm.train_or_load("lora_r16_full", MODEL_ID, train, valid, **HP)
pd.DataFrame(slm.hyperparameter_table(ft_info), columns=["hyperparameter", "value"]).set_index("hyperparameter")
""")

code(r"""
slm.plot_losses({f"LoRA r=16, {ft_info['n_train']} examples": ft_info}, RES / "loss_curve.png",
                title="Qwen3-0.6B + LoRA: loss on the answer tokens")
plt.show()
log = pd.DataFrame([h for h in ft_info["log_history"] if "loss" in h or "eval_loss" in h])
log = log.groupby("step").first().reindex(columns=["epoch", "loss", "eval_loss"]).rename(
    columns={"loss": "train loss", "eval_loss": "validation loss"})
log.round(4).T
""")

# =====================================================================================================
md(r"""
## 4 · After: the fine-tuned model on the same test set

The LoRA update is merged into the weights (in fp32, then cast to the inference precision), so the evaluated model is
exactly the model that is exported to GGUF in §7. Same 80 messages, same prompt, same parser, same greedy decoding.
""")

code(r"""
ft = slm.run_eval("finetuned", MODEL_ID, {"test": test, "tricky": tricky}, adapter=FT_DIR)
m_ft = slm.evaluate(test, ft["test"])
before_after = slm.metrics_frame({"before: base Qwen3-0.6B": m_base, "after: + LoRA fine-tuning": m_ft})
before_after.to_csv(RES / "before_after.csv")
(RES / "before_after.md").write_text(before_after.to_markdown() + "\n", encoding="utf-8")
before_after
""")

code(r"""
errors = [{"id": i, "message": slm.text_of(r)[:90], "correct answer": slm.short(slm.gold_of(r)),
           "fine-tuned model": slm.mark(o["pred"], slm.gold_of(r))}
          for i, (r, o) in enumerate(zip(test, ft["test"])) if o["pred"] != slm.gold_of(r)]
print(f"The fine-tuned model is wrong on {len(errors)} of {len(test)} test messages (✗ = wrong field):")
pd.DataFrame(errors).set_index("id")
""")

# =====================================================================================================
md(r"""
## 5 · Ten prompts side by side

Ten test messages chosen **before training** to cover all six teams, all urgency levels, three escalations, Kazakh
and spam (ids in `SIDE_BY_SIDE` at the top). ✗ marks a wrong field.
""")

code(r"""
side = pd.DataFrame([{"id": i, "message": slm.text_of(test[i]), "correct answer": slm.short(slm.gold_of(test[i])),
                      "base model": slm.mark(base["test"][i]["pred"], slm.gold_of(test[i])),
                      "fine-tuned": slm.mark(ft["test"][i]["pred"], slm.gold_of(test[i]))}
                     for i in SIDE_BY_SIDE]).set_index("id")
(RES / "side_by_side_10.md").write_text(side.to_markdown() + "\n", encoding="utf-8")
side
""")

md(r"""
**The 10 tricky inputs from Assignment 1** (not part of the 80-message metric):
""")

code(r"""
kinds = [r["kind"] for r in load_jsonl(TRICKY_PATH)]
tricky_table = pd.DataFrame([{"kind": k, "message": slm.text_of(r), "correct answer": slm.short(slm.gold_of(r)),
                              "base model": slm.mark(b["pred"], slm.gold_of(r)),
                              "fine-tuned": slm.mark(f["pred"], slm.gold_of(r))}
                             for k, r, b, f in zip(kinds, tricky, base["tricky"], ft["tricky"])])
(RES / "tricky_base_vs_finetuned.md").write_text(tricky_table.to_markdown(index=False) + "\n", encoding="utf-8")
print("all 3 fields right on the tricky inputs: base", sum(o["pred"] == slm.gold_of(r) for r, o in zip(tricky, base["tricky"])),
      "/ 10, fine-tuned", sum(o["pred"] == slm.gold_of(r) for r, o in zip(tricky, ft["tricky"])), "/ 10")
tricky_table
""")

# =====================================================================================================
md(r"""
## 6 · Experiment: how much labelled data does a 0.6B model need?

**Question.** Generating and checking training data costs money and time. How does test accuracy grow with the size
of the training set?

**Setup.** The same hyperparameters as the main run (r = 16, 2 epochs, the same learning-rate schedule), trained on
stratified random subsets of the training set (the same mix of labels) with 100, 200 and 400 examples, plus the main run
with all 819; 0 examples = the base model. Every model is scored on the same 80 test messages. With a fixed number of
epochs a smaller set also means fewer optimizer steps, which is the real trade-off when data is scarce.
""")

code(r"""
runs = {0: (None, m_base)}
for n in EXPERIMENT_SIZES:
    adapter, info = slm.train_or_load(f"lora_r16_n{n}", MODEL_ID, slm.subsample(train, n), valid,
                                      **{**HP, "n_evals": 1, "eval_on_start": False})
    preds = slm.run_eval(f"n{n}", MODEL_ID, {"test": test}, adapter=adapter)["test"]
    runs[n] = (info, slm.evaluate(test, preds))
runs[len(train)] = (ft_info, m_ft)

def last(info, key):
    vals = [h[key] for h in info["log_history"] if key in h] if info else []
    return round(vals[-1], 4) if vals else None

experiment = pd.DataFrame([{
    "training examples": n, "optimizer steps": info["total_steps"] if info else 0,
    "training time, min": round(info["train_time_s"] / 60, 1) if info else 0,
    "last train loss": last(info, "loss"), "validation loss": last(info, "eval_loss"),
    **{name: round(100 * m[key], 1) for key, name in [("exact_match", "all 3 right, %"), ("category_acc", "category, %"),
                                                      ("urgency_acc", "urgency, %"), ("escalate_acc", "escalate, %"),
                                                      ("invalid_rate", "invalid, %")]}}
    for n, (info, m) in sorted(runs.items())]).set_index("training examples")
experiment.to_csv(RES / "experiment_dataset_size.csv")
(RES / "experiment_dataset_size.md").write_text(experiment.to_markdown() + "\n", encoding="utf-8")
experiment
""")

code(r"""
ns = sorted(runs)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for key, name in [("exact_match", "all 3 fields right"), ("category_acc", "category"),
                  ("urgency_acc", "urgency"), ("escalate_acc", "escalate")]:
    ax.plot(range(len(ns)), [100 * runs[n][1][key] for n in ns], marker="o", lw=2.2 if key == "exact_match" else 1.4,
            label=name)
ax.set_xticks(range(len(ns)), ["0\n(base)"] + [str(n) for n in ns[1:]])
ax.set_xlabel("training examples (2 epochs each)")
ax.set_ylabel("accuracy on the 80 test messages, %")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Experiment: training-set size")
fig.tight_layout()
fig.savefig(RES / "experiment_dataset_size.png", dpi=150)
plt.show()
""")

# =====================================================================================================
md(r"""
## 7 · Export: merged model → GGUF → LM Studio

1. **Merge.** W′ = W + (α / r)·B·A for every adapted matrix, computed in fp32 and saved in bf16 as an ordinary
   Hugging Face model. The chat template is changed so that *no thinking* is the default: Qwen3 adds the empty
   `<think></think>` block only when the caller passes `enable_thinking=False`, and LM Studio does not pass it.
   With the changed template every runtime builds exactly the prompt the model was trained on.
2. **Convert** with llama.cpp's `convert_hf_to_gguf.py` to GGUF with 8-bit weights (`q8_0`): ≈ 0.6 GB instead of 1.2 GB,
   with almost no quality loss for a model this small.
3. **Load in LM Studio** and query it through its OpenAI-compatible server on `localhost:1234`.
""")

code(r"""
MERGED = slm.OUTPUTS / "qwen3-0.6b-altyn-router"
if not (MERGED / "model.safetensors").exists():
    slm.merge_and_save(MODEL_ID, FT_DIR, MERGED)
print("merged model:", sorted(p.name for p in MERGED.iterdir()))

LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR") or "llama.cpp")   # a local llama.cpp checkout, or clone one
if not (LLAMA_CPP / "convert_hf_to_gguf.py").exists():
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)],
                   check=True)
GGUF = slm.OUTPUTS / "qwen3-0.6b-altyn-router-q8_0.gguf"
if not GGUF.exists():
    run = subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(MERGED),
                          "--outtype", "q8_0", "--outfile", str(GGUF)], capture_output=True, text=True)
    log = [line.replace(os.getcwd() + "/", "") for line in (run.stdout + run.stderr).splitlines() if "INFO:hf-to-gguf" in line]
    print("\n".join(log[-3:]))
    run.check_returncode()
print(f"\n{GGUF.name}: {GGUF.stat().st_size / 1e6:.0f} MB")
""")

md(r"""
**In LM Studio** (version 0.3+): import the file, load it and start the local server, either in the app
(*My Models → Import*, then *Developer → Start server*) or from a terminal:

```bash
lms import a2_slm/outputs/qwen3-0.6b-altyn-router-q8_0.gguf
lms load qwen3-0.6b-altyn-router-q8_0
lms server start
```

On Colab, download the GGUF file first (*Files* panel → `a2_slm/outputs/` → *Download*). The next cell sends the 80
test messages to the model running in LM Studio. Without a running LM Studio it shows the saved results of the
run on the laptop.
""")

code(r"""
lms_file = RES / "pred_gguf_lmstudio_test.jsonl"
lms_model = slm.lmstudio_model()
if lms_model:
    print("LM Studio model:", lms_model)
    gguf_test = slm.lmstudio_predict(test, lms_model)
    slm.save_predictions(test, gguf_test, lms_file)
elif lms_file.exists():
    print("LM Studio is not running here: saved results of the LM Studio run on the laptop")
    gguf_test = slm.load_predictions(lms_file)
else:
    gguf_test = None
    print("LM Studio is not running and there are no saved LM Studio results")
if gguf_test:
    m_gguf = slm.evaluate(test, gguf_test)
    display(slm.metrics_frame({"fine-tuned, PyTorch bf16": m_ft, "fine-tuned, GGUF q8_0 in LM Studio": m_gguf}))
""")

code(r"""
shot = RES / "lmstudio_screenshot.png"
if shot.exists():
    display(Image(str(shot), width=900))
""")

# =====================================================================================================
md(r"""
## 8 · Bonus: the fine-tuned SLM vs Jev (Assignment 1)

All systems are scored on the same 80 hand-labelled test messages. The Assignment 1 systems (Jev and the LLM
baselines) are paid APIs; their rows are computed from the answers saved in `a1_jev/results/`, so this notebook makes
no API calls. The SLM runs on my own laptop, so its marginal cost is only electricity. To compare it with an API
anyway, the table also prices the measured SLM time at the on-demand price of a cloud T4 GPU (**assumption:** $0.35
per hour, one message at a time; batching would make it far cheaper).

The last column asks the question from Assignment 1: if only decisions with a high enough confidence are made
automatically and the rest go to a human, what share can be automated while keeping ≥ 95 % accuracy? For Jev the
confidence is the minimum of its per-question confidences; for the SLM and the LLMs it is the lowest probability
among the generated answer tokens. The SLM's probabilities come straight from its logits, so they are informative
even with greedy decoding. The DeepSeek API reports them after temperature scaling, so at temperature 0 every token
gets probability 1.0 and those runs have no usable confidence (marked \*).
""")

code(r"""
T4_USD_PER_HOUR = 0.35   # assumption, see above

from a1_jev import analyze   # the Assignment 1 metrics code, so both notebooks compute the numbers the same way

def row(system, m, latency, cost_1000, auto_share, note=""):
    return {"system": system, "all 3 right": f"{100 * m['exact_match']:.1f} %",
            "category": f"{100 * m['category_acc']:.1f} %", "urgency": f"{100 * m['urgency_acc']:.1f} %",
            "escalate": f"{100 * m['escalate_acc']:.1f} %", "invalid": f"{100 * m['invalid_rate']:.1f} %",
            "latency, s": f"{latency:.2f}", "cost per 1,000 decisions": cost_1000,
            "auto-decided at ≥95 % accuracy": "—" if auto_share is None else f"{100 * auto_share:.0f} %{note}"}

table = []
a1_runs = analyze.load_runs()   # every Assignment 1 system with saved answers, Jev first
if "jev" not in a1_runs:
    print("No Jev answers from Assignment 1 yet: its row appears here once a1_jev/results/raw_jev.jsonl exists.")
for name, a1 in a1_runs.items():
    m = analyze.system_metrics(a1)
    table.append(row(analyze.label(name) + ", API", m, m["avg_latency_s"], f"${m['cost_per_1000_usd']:.4f}",
                     m["auto_share_95"], "" if m["confidence_kind"] == "informative" else " *"))

def slm_coverage(outs):
    cov = auto_share_at([o["min_token_prob"] for o in outs], [o["pred"] == slm.gold_of(r) for r, o in zip(test, outs)])
    return cov["auto_share"] if cov else 0.0

t4 = 1000 * m_ft["avg_latency_s"] / 3600 * T4_USD_PER_HOUR
table.append(row(f"Qwen3-0.6B + LoRA, PyTorch on {ft_info['device_name']}", m_ft, m_ft["avg_latency_s"],
                 f"$0 on own laptop (≈ ${t4:.3f} on a cloud T4)", slm_coverage(ft["test"])))
if gguf_test:
    table.append(row("Qwen3-0.6B + LoRA, GGUF q8_0 in LM Studio", m_gguf, m_gguf["avg_latency_s"],
                     "$0 on own laptop", None))
table.append(row("Qwen3-0.6B base (no fine-tuning)", m_base, m_base["avg_latency_s"], "$0 on own laptop",
                 slm_coverage(base["test"])))
bonus = pd.DataFrame(table).set_index("system")
note = ("\\* no usable confidence: every answer has probability 1.0, so the system can only decide all messages "
        "or none, and all are decided automatically only if accuracy is ≥ 95 %.")
(RES / "bonus_vs_jev.md").write_text(bonus.to_markdown() + "\n\n" + note + "\n", encoding="utf-8")
display(bonus)
print(note.replace("\\", ""))
""")

# =====================================================================================================
md(r"""
## 9 · Conclusions

_Written after the run; see the README for the final text._
""", )


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    }
    nbf.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK.relative_to(HERE.parent)}: {len(CELLS)} cells")


if __name__ == "__main__":
    build()
