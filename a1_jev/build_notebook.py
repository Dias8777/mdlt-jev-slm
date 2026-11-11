"""Write the Assignment 1 notebook a1_jev/a1_jev_routing.ipynb (cells without outputs).

    python -m a1_jev.build_notebook

The notebook is kept as code here so that its text and cells are easy to review and diff. It needs no API key:
without one it replays the API answers saved in a1_jev/results/raw_*.jsonl.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "a1_jev_routing.ipynb"

CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# =====================================================================================================
md(r"""
# Assignment 1 · Jev (TypeSafe System One) vs an LLM for routing customer messages

**Decision task.** *Altyn Mobile* is a fictional mobile operator in Kazakhstan. Customers write in Russian, Kazakh,
English or a mix of them. Every message gets three decisions:

| field | values | meaning |
|---|---|---|
| `category` | `billing` · `network` · `tariff` · `sim` · `account` · `other` | which team solves the main problem |
| `urgency` | `low` · `medium` · `high` | how fast it must be handled |
| `escalate` | `true` · `false` | send to a senior agent: fraud or a security incident, a threat to leave / sue / complain, repeated contact |

Example: «Украли телефон в автобусе! Срочно заблокируйте мой номер» → `sim · high · escalate`

| § | what |
|---|---|
| 1 | System One vs LLM (½ page) |
| 2 | the Jev workflow: 4 typed questions combined in code with thresholds, diagram |
| 3 | the test data: 80 hand-labelled messages + 10 tricky ones |
| 4 | one decision step by step |
| 5 | running Jev and two DeepSeek models (JSON output) on the same messages |
| 6 | results: accuracy, latency, cost per 1,000 decisions, % invalid |
| 7 | accuracy vs confidence threshold: what share can be automated at ≥ 95 % accuracy |
| 8 | the 10 tricky inputs with every system's output |
| 9 | conclusions: Jev vs LLM |

**How to run.** Colab: *Runtime → Run all*. **No API key is needed**: the notebook replays the API answers saved in
`a1_jev/results/raw_*.jsonl` and recomputes every number from them. To call the APIs yourself, add
a Jev key (`OPENROUTER_API_KEY`, `TYPESAFE_API_KEY` or `AI_GATEWAY_API_KEY`, see `.env.example`) and `DEEPSEEK_API_KEY`
as Colab secrets (the key icon on the left) or put them into a `.env` file, and set `RUN_API = True` in §5.
Keys are read at run time by `common/keys.py` and never appear in the notebook.
""")

md((HERE / "system_one_vs_llm.md").read_text(encoding="utf-8").replace("## System One vs LLM", "## 1 · System One vs LLM"))

code(r"""
# Setup. On Colab: get the code, the test data and the saved API answers.
import os, sys, subprocess

if "google.colab" in sys.modules:
    if not os.path.isdir("/content/mdlt-jev-slm"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/Dias8777/mdlt-jev-slm.git",
                        "/content/mdlt-jev-slm"], check=True)
    os.chdir("/content/mdlt-jev-slm")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabulate"], check=True)
else:   # local run: the notebook lives in a1_jev/, the code is imported from the repository root
    while not os.path.isdir("common"):
        os.chdir("..")
sys.path.insert(0, os.getcwd())
""")

code(r"""
import inspect, json
from collections import Counter

import pandas as pd
from IPython.display import Image, Markdown, display

from a1_jev import analyze, workflow
from common.keys import get_key
from common.task import FIELDS, TEST_PATH, TRICKY_PATH, URGENCY, load_jsonl

pd.set_option("display.max_colwidth", 120)
RES = analyze.RESULTS
HAVE_JEV_KEY = any(get_key(b["key"], required=False) for b in workflow.BACKENDS.values())
HAVE_DEEPSEEK_KEY = bool(get_key("DEEPSEEK_API_KEY", required=False))
print(f"Jev model: {workflow.JEV_MODEL} | Jev key found: {HAVE_JEV_KEY} | DeepSeek key found: {HAVE_DEEPSEEK_KEY}")
print("saved API answers:", sorted(p.name for p in RES.glob("raw_*.jsonl")))
""")

# =====================================================================================================
md(r"""
## 2 · The Jev workflow

One Jev request per message asks **four typed questions** about the same `state = {"customer_message": text}`.
The rules in the criteria are the same labelling guide that was used to label the test set by hand
(`common/task.py`), split into the four questions.
""")

code(r"""
def criteria_md(q):
    c = q["criteria"]
    if isinstance(c, dict):
        return "<br>".join(f"**{k}**: {v}" for k, v in c.items())
    return "<br>".join(f"**{i}** = {URGENCY[i]}: {v}" for i, v in enumerate(c))

rows = ["| question | type | instructions | options / criteria |", "|---|---|---|---|"]
for name, q in workflow.QUESTIONS.items():
    rows.append(f"| `{name}` | `{q['type']}` | {q['instructions']} | {criteria_md(q)} |")
display(Markdown("\n".join(rows)))
""")

md(r"""
**Typed answers → decision.** Plain Python turns the four answers into the three decisions, with two thresholds
(`TAU_SECURITY`, `TAU_ESCALATE`, both 0.5) and one more rule from the labelling guide: a security incident is always
`high` urgency. Each answer has a confidence between 0 and 1, computed the way TypeSafe recommends:

* `noul` with probability *p*: |2*p* − 1| (0 at *p* = 0.5, 1 at *p* = 0 or 1);
* `choice`: how far the top probability is above a uniform guess, (p<sub>max</sub> − 1/n) / (1 − 1/n);
* `score`: high when the probability mass is concentrated on one level.

The decision is only as reliable as its weakest part, so its confidence is the **minimum** of the three. §7 uses
this number: decisions above a threshold are made automatically, the rest go to a human.
""")

code(r"""
print(inspect.getsource(workflow.decide))
""")

code(r"""
display(Image(str(RES / "workflow.png"), width=1100))
""")

# =====================================================================================================
md(r"""
## 3 · Test data

`data/test_tickets.jsonl`: **80 messages written and labelled by hand** (source with labels: `data/make_test_sets.py`)
in Russian, Kazakh, a Kazakh–Russian mix and English, from three-word messages to long complaints. Before any
model was run, every label was re-checked against the written guide (5 labels fixed, marked `# audit:`).
`data/tricky_tickets.jsonl`: **10 tricky inputs**: sarcasm, prompt injection, two problems in one message, Kazakh in
Latin letters, typos, emoji only, a negation. They are reported separately (§8), not in the 80-message metrics.
""")

code(r"""
test, tricky = load_jsonl(TEST_PATH), load_jsonl(TRICKY_PATH)
for f in FIELDS:
    c = Counter(str(r[f]).lower() for r in test)
    print(f"{f:>9}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
print("tricky kinds:", ", ".join(r["kind"] for r in tricky))
pd.DataFrame(test)[["id", "text", *FIELDS]].sample(6, random_state=1).set_index("id")
""")

# =====================================================================================================
md(r"""
## 4 · One decision step by step

Test message 49: a SIM card re-issued without the customer's consent. With a Jev key the cell calls the API now;
without one it shows the answer saved during the run in §5.
""")

code(r"""
DEMO_ID = 49
demo = next(r for r in test if r["id"] == DEMO_ID)
print("message:", demo["text"])
print("request body:", json.dumps({"model": workflow.JEV_MODEL, "state": {"customer_message": demo["text"]},
                                   "questions": {k: {"type": q["type"], "...": "..."} for k, q in workflow.QUESTIONS.items()}},
                                  ensure_ascii=False))
answers = None
if HAVE_JEV_KEY:
    try:
        res = workflow.ask_jev(demo["text"])
        answers = res["answers"]
        print(f"\nlive call via {res['backend']}: {res['latency_s']:.2f} s, ${res['cost']:.7f}, {res['usage']}, "
              f"model {res['model']}")
    except RuntimeError as e:
        print(f"\nlive call failed ({str(e)[:150]}), showing the saved answer instead")
if answers is None and (RES / "raw_jev.jsonl").exists():
    saved = next(r for r in load_jsonl(RES / "raw_jev.jsonl") if r["set"] == "test" and r["id"] == DEMO_ID)
    answers = saved["raw"]
    print(f"\nsaved answer from {saved['time_utc']}: {saved['latency_s']:.2f} s, ${saved['cost']:.7f}, {saved['usage']}")
if answers:
    print("\nJev's typed answers:\n" + json.dumps(answers, ensure_ascii=False, indent=1))
    d = workflow.decide(answers)
    print("\ndecision:", {k: d[k] for k in FIELDS}, "| confidence", round(d["confidence"], 3))
    print("correct:  ", {k: demo[k] for k in FIELDS})
else:
    print("no Jev key and no saved Jev answers")
""")

# =====================================================================================================
md(r"""
## 5 · Running every system on the same messages

`python -m a1_jev.run <system>` sends the 80 test and 10 tricky messages **one at a time** (so the latency is the
latency of one decision, network from Almaty included) and appends every answer with its latency, tokens and cost to
`a1_jev/results/raw_<system>.jsonl`. Messages already in the file are skipped, so re-running costs nothing.

| system | how it is asked | output |
|---|---|---|
| **Jev 1.14** (TypeSafe's System One API, called through OpenRouter) | the 4 typed questions of §2 | probabilities → `decide()` |
| **DeepSeek V4 Flash** (DeepSeek API) | system prompt = the labelling guide; JSON mode; temperature 0; thinking off | a JSON object + token probabilities |
| **DeepSeek V4 Pro** (DeepSeek API) | the same | a JSON object + token probabilities |
| **DeepSeek V4 Flash, T = 1** | the same, but temperature 1 | a JSON object + informative token probabilities |

The LLMs get the whole labelling guide as their system prompt and must answer with
`{"category": …, "urgency": …, "escalate": …}`. Their output goes through a strict parser (`common/task.py:
parse_answer`): text that is not one JSON object with allowed labels counts as **invalid** and as wrong on every
field. The LLMs are also asked for the probability of every generated token (`logprobs`); the probability of
the **least certain token** of the answer is used as the LLM's confidence in §7. DeepSeek reports these
probabilities *after* temperature scaling: at temperature 0 the chosen token always gets probability 1.0, so the
two main runs carry no usable confidence. The extra Flash run at temperature 1 returns the model's own
probabilities, at a price: its answer is then sampled from them and can change from one run to the next.

Cost: Jev is priced at $0.042 per 1 M input tokens, output free (the exact charge when the API reports one).
DeepSeek is priced from the returned token counts at its off-peak list price (Flash $0.15 / 1 M input,
$0.003 cached, $0.60 output; Pro $0.66 / $0.022 / $1.98; twice that in DeepSeek's peak hours). DeepSeek caches the
repeated system prompt, so most input tokens are billed at the cached price.
""")

code(r"""
RUN_API = False   # True: call the APIs for every message that is not in the saved files yet (needs the keys)

if RUN_API:
    jobs = ([["jev"]] if HAVE_JEV_KEY else []) + \
           ([["llm", "deepseek-flash"], ["llm", "deepseek-v4-pro"],
             ["llm", "deepseek-flash", "--temperature", "1", "--name", "deepseek-flash-t1"]] if HAVE_DEEPSEEK_KEY else [])
    for job in jobs:
        out = subprocess.run([sys.executable, "-m", "a1_jev.run", *job], capture_output=True, text=True)
        print("\n".join(out.stdout.strip().splitlines()[:1] + out.stdout.strip().splitlines()[-1:]), out.stderr[-500:])
for f in sorted(RES.glob("raw_*.jsonl")):
    rows = load_jsonl(f)
    times = sorted(r["time_utc"] for r in rows)
    print(f"{f.name}: {len(rows)} answers, {times[0]} … {times[-1]}")
""")

# =====================================================================================================
md(r"""
## 6 · Results on the 80 hand-labelled test messages

*Exact match* = all three fields right. *Escalate recall* = share of the 15 messages that must be escalated which
were escalated. *% invalid* = outputs that could not be used (not JSON, or a label outside the allowed set).
Latency is wall-clock time of one API request from Almaty. Cost per 1,000 decisions = 1,000 × the average cost
of one test message. *Auto-decided at ≥ 95 % accuracy* is explained in §7.
""")

code(r"""
runs = analyze.load_runs()
metrics = analyze.write_table(runs)
display(Markdown((RES / "results_table.md").read_text(encoding="utf-8")))
""")

# =====================================================================================================
md(r"""
## 7 · Accuracy vs confidence threshold

A business does not have to let the machine decide every message. Jev's decision confidence (the weakest of its
three answers) allows a rule: **decide automatically when confidence ≥ t, otherwise give the message to a human.**
Moving t from 1 down to 0 trades the share of automatic decisions against their accuracy. The target here is
**≥ 95 % exact-match accuracy** on the automatic part, and the chosen t is the one that keeps the largest share
automatic while meeting it (messages with equal confidence cannot be separated by a threshold, so they stay
together). Every system gets its own threshold, because the confidences are on different scales. For the LLMs the
confidence is the probability of the least certain token of their JSON answer; the right plot compares all systems.
At temperature 0 that probability is 1.0 for every answer, so those two runs are single points: they can only
decide everything (the table below shows the evidence).

The threshold is chosen on the same 80 messages it is reported on, so the numbers are optimistic; with more data it
would be chosen on a separate validation set.
""")

code(r"""
if "jev" in runs:
    best = analyze.plot_threshold(runs, metrics)
    display(Image(str(RES / "jev_threshold.png"), width=1100))
    if best:
        print(f"best threshold t = {best['threshold']:.3f}: {best['n_auto']} of 80 messages "
              f"({best['auto_share']:.0%}) decided automatically with {best['auto_accuracy']:.1%} accuracy, "
              f"{best['human_share']:.0%} sent to a human")
    else:
        print("no threshold reaches the 95 % target")
    sw = pd.DataFrame(analyze.sweep(runs["jev"])).set_index("threshold")
    display(sw.loc[[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]].style.format(
        {"auto_share": "{:.0%}", "human_share": "{:.0%}", "auto_accuracy": "{:.1%}"}))
else:
    best = None
    print("no Jev answers yet")
""")

code(r"""
# How well does each system's confidence separate its right answers from its wrong ones?
rows = []
for name, run in runs.items():
    test = analyze.by_set(run, "test")
    right = [r["confidence"] for r in test if r["pred"] == r["gold"]]
    wrong = [r["confidence"] for r in test if r["pred"] != r["gold"]]
    m = metrics[name]
    rows.append({"system": analyze.label(name), "different confidence values": len({r["confidence"] for r in test}),
                 "median conf., right answers": analyze.quantile(right, 0.5),
                 "median conf., wrong answers": analyze.quantile(wrong, 0.5) if wrong else float("nan"),
                 "exact match": m["exact_match"], "auto-decided at ≥ 95 %": m["auto_share_95"]})
display(pd.DataFrame(rows).set_index("system").style.format(
    {"median conf., right answers": "{:.4f}", "median conf., wrong answers": "{:.4f}", "exact match": "{:.1%}",
     "auto-decided at ≥ 95 %": "{:.0%}"}, na_rep="—"))
""")

code(r"""
if "jev" in runs:
    jev_test = analyze.by_set(runs["jev"], "test")
    wrong = [{"id": r["id"], "message": r["text"][:90], "correct": analyze.fmt_pred(r["gold"]),
              "Jev": analyze.fmt_pred(r["pred"], r["gold"]), "confidence": round(r["confidence"], 3),
              "routed": "auto" if best and r["confidence"] >= best["threshold"] else "→ human"}
             for r in jev_test if r["pred"] != r["gold"]]
    print(f"Jev is wrong on {len(wrong)} of {len(jev_test)} test messages; with the threshold above, "
          f"{sum(w['routed'] == '→ human' for w in wrong)} of them go to a human (bold✗ = wrong field):")
    if wrong:
        display(pd.DataFrame(wrong).set_index("id").sort_values("confidence"))
""")

# =====================================================================================================
md(r"""
## 8 · The 10 tricky inputs

Sarcasm, a Kazakh–Russian mix, heavy typos, two prompt-injection attempts, two problems in one message, Kazakh in
Latin letters, English, two emoji and a negation. **bold✗** marks a wrong field. For every system with a usable
confidence (Jev, Flash at T = 1), the confidence and whether that system's threshold from §7 would let it decide
alone are shown in brackets.
""")

code(r"""
analyze.write_tricky(runs, metrics)
display(Markdown((RES / "tricky_cases.md").read_text(encoding="utf-8")))
""")

# =====================================================================================================
md(r"""
## 9 · Conclusions: Jev vs LLM

* **Accuracy: a tie with the best LLM.** On the 80 hand-labelled messages Jev gets all three fields right on 93.8 %,
  the same as DeepSeek V4 Flash and more than DeepSeek V4 Pro (87.5 %). It never misses a message that must be
  escalated (escalate accuracy and recall 100 %), and it can never return an invalid answer, because every answer
  has the declared type. Four of its five errors are about urgency, the field where the guide leaves the most room
  for judgement.
* **Speed and cost.** One Jev decision takes 0.44 s on average (p95 0.53 s), about three times faster than the LLMs
  (1.2–1.5 s), because nothing is generated. It costs $0.039 per 1,000 decisions, the same as Flash ($0.036) and half
  of Pro ($0.075). The whole Jev run over 90 messages cost $0.0035.
* **Confidence is the real difference.** Jev's probabilities are informative: with the threshold t = 0.58 it decides
  94 % of the messages on its own at 96 % accuracy and sends 6 % to a person. DeepSeek at temperature 0 reports
  probability 1.0 for everything, so it can only decide all messages or none; at temperature 1 its probabilities
  become useful (91 % automatic), but its answers are then sampled and its exact match drops to 88.8 %.
* **Tricky inputs: Jev is weaker, but it knows it.** All three LLM runs got 9 of 10 right, Jev 7 of 10: it failed
  on Kazakh in Latin letters, on "😡😡😡 баланс" and on the Kazakh–Russian mix, where an LLM's knowledge of languages
  helps. All three wrong answers had confidence 0.20–0.44, below the threshold, so in the workflow they go to a
  person and no wrong decision is made automatically. Both prompt injections had no effect on Jev, since the message
  is only the `state` it reads, never an instruction.
* **What to use.** For this routing task Jev is the better first-line tool: as accurate as the LLM, three times
  faster, with a confidence that tells when a person must look, and with the decision policy in plain code. The
  uncertain messages (about 6 %) can go to a person or to an LLM. One caveat: the threshold was chosen on the same 80
  messages it is reported on, and the tricky set has only 10 messages, so these numbers should be confirmed on new,
  real messages.
""")


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    nbf.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK.relative_to(HERE.parent)}: {len(CELLS)} cells")


if __name__ == "__main__":
    build()
