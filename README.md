# Jev vs LLM, and a fine-tuned 0.6B router

MDLT, AITU · Assignment 1 (TypeSafe Jev, a "System One" model, vs an LLM) and Assignment 2 (fine-tuning a small
language model). Both assignments solve **the same decision task** on **the same hand-labelled test set**, so every
number below is directly comparable.

| | notebook | Colab |
|---|---|---|
| **Assignment 1** · Jev vs LLM | [`a1_jev/a1_jev_routing.ipynb`](a1_jev/a1_jev_routing.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dias8777/mdlt-jev-slm/blob/main/a1_jev/a1_jev_routing.ipynb) |
| **Assignment 2** · Qwen3-0.6B + LoRA | [`a2_slm/a2_finetune_qwen3.ipynb`](a2_slm/a2_finetune_qwen3.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dias8777/mdlt-jev-slm/blob/main/a2_slm/a2_finetune_qwen3.ipynb) |

The notebooks are saved with their outputs. **There are no API keys in this repository**: keys are read at run time
from environment variables, Colab secrets or a git-ignored `.env` file (see [`.env.example`](.env.example)).
The Assignment 1 notebook runs without any key, because it replays the API answers saved in `a1_jev/results/`.

## The decision task

*Altyn Mobile* is a fictional mobile operator in Kazakhstan. Customers write to support in Russian, Kazakh,
English or a mix. Every message needs three decisions:

| field | values | meaning |
|---|---|---|
| `category` | `billing` · `network` · `tariff` · `sim` · `account` · `other` | which team must solve the main problem |
| `urgency` | `low` · `medium` · `high` | how fast it must be handled |
| `escalate` | `true` · `false` | send to a senior agent: fraud or a security incident, a threat to leave / sue / complain, repeated contact |

«Украли телефон в автобусе! Срочно заблокируйте мой номер» → `{"category": "sim", "urgency": "high", "escalate": true}`

The written labelling guide in [`common/task.py`](common/task.py) is the single source of truth: the same rules
label the test set, form the LLM prompt, the SLM system prompt and the training-data pipeline, and are split into
Jev's typed questions.

**Test set:** [`data/test_tickets.jsonl`](data/test_tickets.jsonl), **80 messages written and labelled by hand**
(13–14 per team; 15 must be escalated) before any model was run or any training data existed. Every label was
re-checked against the guide once (5 fixes, marked `# audit:` in [`data/make_test_sets.py`](data/make_test_sets.py)).
**Tricky inputs:** [`data/tricky_tickets.jsonl`](data/tricky_tickets.jsonl), 10 messages with sarcasm, prompt
injection, two problems, Kazakh in Latin letters, typos, emoji only and a negation.

---

## Assignment 1 · Jev (TypeSafe System One) vs an LLM

### System One vs LLM

Kahneman's *System 1* is the fast, automatic part of thinking: we recognise a face or hear that a reply is rude
without reasoning step by step. *System 2* is slow and deliberate. TypeSafe borrows the name: **Jev is a "System One"
model.** It does not write text. A request gives it some content (`state`) and a set of **typed questions**, and in
one pass it returns, for every question, a probability distribution of the declared type: `noul` (the probability
that a yes/no statement is true), `choice` (a distribution over named options) or `score` (a distribution over
ordered levels). So the answer always has the declared shape, every answer comes with a confidence, and because
nothing is generated, the price depends on the input only (output tokens are free) and the response time does not
grow with the length of an answer.

A general **LLM** is closer to System 2. It generates tokens one at a time, so it can follow any instruction, reason
through several steps, extract or rewrite text and explain itself. For a decision task this flexibility has costs:
the answer has to be parsed (it can be broken JSON or an invented label), the confidence an LLM states in words is
not a calibrated probability (token probabilities are a better signal, where an API returns them), and every
generated token adds latency and money.

The two also split the work differently. With an LLM the whole decision policy sits inside a prompt, and the model
both reads the message and decides. With Jev the model only *perceives*: it answers narrow questions about the
message. The *policy* (which answers lead to which action, which thresholds, when a person must look) is ordinary
code that can be read, unit-tested and changed without re-prompting or retraining. Jev cannot do what needs several
reasoning steps, calculation or free text; an LLM is the right tool there. For high-volume routing, triage and
moderation, where the output is one of a few known options and the business must know when the machine is unsure,
a System One model fits better.

### The workflow: 4 typed questions + code with thresholds

![workflow](a1_jev/results/workflow.png)

One Jev request per message ([`a1_jev/workflow.py`](a1_jev/workflow.py)), `state = {"customer_message": text}`:

| question | type | what it asks |
|---|---|---|
| `category` | `choice`, 6 options | which team must solve the main problem (each option carries its definition from the guide) |
| `urgency` | `score`, levels 0–2 | 0 = low, 1 = medium, 2 = high, each level with its definition |
| `security_incident` | `noul` | P(fraud or a security incident: lost or stolen phone, hijacked account, SIM re-issued without consent, data leak, money stolen through the number) |
| `pressure` | `noul` | P(threat to leave, sue or complain to a regulator / social media, or "I have already written several times") |

The code that turns the typed answers into a decision:

```python
category   = the option Jev chose
urgency    = the most likely level;  if P(security_incident) >= 0.5: urgency = "high"   # rule from the guide
escalate   = max(P(security_incident), P(pressure)) >= 0.5
confidence = min(conf(category), conf(urgency), |2·P(escalate) − 1|)    # the weakest link
route      = automatic if confidence >= t, otherwise a human decides
```

The LLM baselines get the full labelling guide as their system prompt and must return the JSON object
(JSON mode, temperature 0, thinking off). Their output goes through a strict parser: anything that is not one JSON
object with allowed labels counts as **invalid** and as wrong. The LLMs are also asked for token probabilities
(`logprobs`), and the probability of the least certain token is used as their confidence. DeepSeek reports these
probabilities after temperature scaling, so at temperature 0 every token has probability 1.0; a third run, Flash at
temperature 1, shows what the model's own probabilities can do.

### Results on the 80 hand-labelled test messages

<!-- A1-RESULTS -->

### Accuracy vs confidence threshold

<!-- A1-THRESHOLD -->

### The 10 tricky inputs

<!-- A1-TRICKY -->

### Jev vs LLM

<!-- A1-CONCLUSION -->

---

## Assignment 2 · Fine-tuning Qwen3-0.6B with LoRA

**Model: Qwen3-0.6B** (Alibaba, Apache-2.0). It is the smallest current open model trained on Russian and Kazakh
(Qwen3 covers 119 languages), and its *non-thinking* mode answers directly, which is what a router needs. At
0.6 B parameters LoRA training fits on a free Colab T4 and on a 16 GB laptop, and the 8-bit GGUF file is ≈ 0.6 GB;
the question is how far a *tiny* model gets after fine-tuning on the task of Assignment 1 (same task → +10 bonus).

### Data

* [`a2_slm/DATASET_CARD.md`](a2_slm/DATASET_CARD.md): how the data was generated, cleaned, re-labelled and split,
  with all statistics and limitations.
* [`a2_slm/data/sample_20.md`](a2_slm/data/sample_20.md): **20-row sample** with how each label was decided;
  the same rows exactly as trained: [`sample_20.jsonl`](a2_slm/data/sample_20.jsonl).
* **819 train / 89 validation** pairs in TRL's chat JSONL format, generated by `deepseek-flash` in 9 language styles,
  deduplicated (exact + TF-IDF near-duplicates), every message too similar to a test message removed (cosine ≥ 0.60),
  then **re-labelled blind** by a second model with a third vote as a tie-breaker (448 of 908 labels changed;
  34 messages without a majority dropped). Cost of the whole dataset: $0.24.
* **Held-out test set: the 80 hand-written messages of Assignment 1** (+ the 10 tricky ones), never seen in training.

<!-- A2-RESULTS -->

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your own keys into .env (it is git-ignored)
```

| step | command | needs |
|---|---|---|
| A1: ask Jev | `python -m a1_jev.run jev` | `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY` or `AI_GATEWAY_API_KEY` |
| A1: LLM baselines | `python -m a1_jev.run llm deepseek-flash`, `python -m a1_jev.run llm deepseek-v4-pro`, `python -m a1_jev.run llm deepseek-flash --temperature 1 --name deepseek-flash-t1` | `DEEPSEEK_API_KEY` |
| A1: tables and plots | `python -m a1_jev.analyze` | nothing (reads `a1_jev/results/raw_*.jsonl`) |
| A2: dataset | `python -m a2_slm.make_dataset` | `DEEPSEEK_API_KEY`, or nothing: every API answer is cached in `a2_slm/data/cache/` |
| A2: notebook | run `a2_slm/a2_finetune_qwen3.ipynb` | a GPU (Colab T4 or Apple Silicon) |
| notebooks from code | `python -m a1_jev.build_notebook`, `python -m a2_slm.build_notebook` | `nbformat` |

## Repository layout

```
common/          task.py (labels, guide, prompt, parser) · metrics.py · llm.py (DeepSeek/OpenRouter client) · keys.py
data/            the 80 hand-labelled test messages and 10 tricky inputs (+ the script that writes them)
a1_jev/          workflow.py (Jev questions + decision code) · run.py · analyze.py · notebook · results/
a2_slm/          make_dataset.py · dataset_report.py · slm.py (training, evaluation, export) · notebook · data/ · results/
```
