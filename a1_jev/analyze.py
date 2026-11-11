"""Assignment 1 analysis: results table, confidence-threshold plot, tricky cases, workflow diagram.

    python -m a1_jev.analyze

Reads every a1_jev/results/raw_<system>.jsonl written by a1_jev/run.py and writes into a1_jev/results/:
    results_table.md / .csv   accuracy, macro-F1, escalate recall, % invalid, latency, cost per 1,000 decisions,
                              share that can be decided automatically at >= 95 % accuracy
    threshold_sweep.csv       Jev: accuracy of the auto-decided part and share sent to a human, per threshold
    jev_threshold.png         the same as a plot, plus accuracy vs automation for every system
    tricky_cases.md           the 10 tricky inputs with every system's output
    workflow.png              the workflow diagram
Jev's decisions are recomputed from the saved typed answers, so changing a threshold in workflow.py needs
no new API calls. Confidence: Jev = its weakest typed answer (workflow.decide); an LLM = the probability of its
least certain generated token (from the API's logprobs). DeepSeek reports these probabilities after temperature
scaling, so at temperature 0 they are all 1.0: such a run has no usable confidence and can only decide everything.
"""
import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from a1_jev import workflow
from common.metrics import auto_share_at, score
from common.task import FIELDS, load_jsonl

RESULTS = Path(__file__).resolve().parent / "results"
TARGET = 0.95
LABELS = {"jev": "Jev 1.14 (4 typed questions)", "deepseek-flash": "DeepSeek V4 Flash (JSON, T = 0)",
          "deepseek-v4-pro": "DeepSeek V4 Pro (JSON, T = 0)", "deepseek-flash-t1": "DeepSeek V4 Flash (JSON, T = 1)"}


def label(name):
    return LABELS.get(name, name)


def load_runs():
    runs = {}
    for f in sorted(RESULTS.glob("raw_*.jsonl")):
        name = f.stem[len("raw_"):]
        rows = load_jsonl(f)
        if name == "jev":   # re-derive decisions from the typed answers with the current thresholds
            for r in rows:
                d = workflow.decide(r["raw"])
                r["pred"], r["confidence"], r["decision"] = {k: d[k] for k in FIELDS}, d["confidence"], d
        runs[name] = rows
    # Jev first, then the LLMs
    return dict(sorted(runs.items(), key=lambda kv: (kv[0] != "jev", kv[0])))


def by_set(rows, name):
    return sorted((r for r in rows if r["set"] == name), key=lambda r: r["id"])


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def input_tokens(r):
    u = r.get("usage") or {}
    return u.get("input_tokens", u.get("prompt_tokens", 0))


def confidence_kind(rows):
    """'informative' if the confidences can rank the test answers, 'constant' if they are all equal
    (DeepSeek at temperature 0 gives 1.0 every time), 'none' if the system returns no confidence."""
    conf = [r.get("confidence") for r in by_set(rows, "test")]
    if any(c is None for c in conf):
        return "none"
    return "constant" if len(set(conf)) == 1 else "informative"


def has_confidence(rows):
    return confidence_kind(rows) == "informative"


def system_metrics(rows):
    test = by_set(rows, "test")
    m = score([r["gold"] for r in test], [r["pred"] for r in test], statuses=[r["status"] for r in test],
              latencies=[r["latency_s"] for r in test])
    lat = [r["latency_s"] for r in test]
    m["p50_latency_s"], m["p95_latency_s"] = quantile(lat, 0.5), quantile(lat, 0.95)
    m["cost_per_1000_usd"] = 1000 * sum(r["cost"] for r in test) / len(test)
    m["input_tokens_avg"] = sum(input_tokens(r) for r in test) / len(test)
    m["confidence_kind"] = kind = confidence_kind(rows)
    m["coverage"] = best_threshold(rows) if kind == "informative" else None
    if kind == "informative":
        m["auto_share_95"] = m["coverage"]["auto_share"] if m["coverage"] else 0.0
    else:   # nothing to rank by: all messages are decided automatically or none
        m["auto_share_95"] = 1.0 if m["exact_match"] >= TARGET else 0.0
    return m


def sweep(rows):
    """For each confidence threshold t: decide automatically when confidence >= t, otherwise a human decides."""
    test = by_set(rows, "test")
    out = []
    for i in range(101):
        t = i / 100
        auto = [r for r in test if r["confidence"] >= t]
        acc = sum(r["pred"] == r["gold"] for r in auto) / len(auto) if auto else float("nan")
        out.append({"threshold": t, "n_auto": len(auto), "auto_share": len(auto) / len(test),
                    "human_share": 1 - len(auto) / len(test), "auto_accuracy": acc})
    return out


def best_threshold(rows, target=TARGET):
    """Largest automatic share whose accuracy is still >= target (sort by confidence, grow the auto set)."""
    test = by_set(rows, "test")
    return auto_share_at([r["confidence"] for r in test], [r["pred"] == r["gold"] for r in test], target)


def coverage_curve(rows):
    """(share decided automatically, accuracy of that part) as the threshold moves down through the confidences."""
    test = sorted(by_set(rows, "test"), key=lambda r: -r["confidence"])
    pts, ok = [], 0
    for k, r in enumerate(test, 1):
        ok += r["pred"] == r["gold"]
        if k < len(test) and test[k]["confidence"] == r["confidence"]:
            continue   # equal confidences cannot be separated by a threshold
        pts.append((k / len(test), ok / k))
    return pts


def cell(value, fmt):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return fmt.format(value)


def fmt_pred(p, gold=None):
    if p is None:
        return "**invalid**"
    parts = []
    for f in FIELDS:
        v = p[f] if f != "escalate" else ("escalate" if p[f] else "no esc.")
        wrong = gold is not None and p[f] != gold[f]
        parts.append(f"**{v}**✗" if wrong else str(v))
    return " · ".join(parts)


def write_table(runs):
    cols = [("exact_match", "Exact match (all 3)", "{:.1%}"), ("category_acc", "Category acc", "{:.1%}"),
            ("category_macro_f1", "Category macro-F1", "{:.3f}"), ("urgency_acc", "Urgency acc", "{:.1%}"),
            ("escalate_acc", "Escalate acc", "{:.1%}"), ("escalate_recall", "Escalate recall", "{:.1%}"),
            ("invalid_rate", "% invalid", "{:.1%}"), ("avg_latency_s", "Avg latency, s", "{:.2f}"),
            ("p95_latency_s", "p95 latency, s", "{:.2f}"), ("cost_per_1000_usd", "Cost / 1,000 decisions", "${:.4f}"),
            ("input_tokens_avg", "Input tokens / decision", "{:.0f}"),
            ("auto_share_95", "Auto-decided at ≥ 95 % accuracy", "{:.0%}")]
    metrics = {name: system_metrics(rows) for name, rows in runs.items()}
    lines = ["| System | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in metrics.items():
        cells = [cell(m[c[0]], c[2]) for c in cols]
        if m["confidence_kind"] != "informative":
            cells[-1] += "\\*"
        lines.append(f"| {label(name)} | " + " | ".join(cells) + " |")
    n = next(iter(metrics.values()))["n"]
    lines.append(f"\nTest set: {n} hand-labelled messages. Latency = one request at a time from Almaty "
                 "(network included). % invalid = output that is not valid JSON with allowed labels. "
                 "Cost: Jev at $0.042 per 1M input tokens (output free); DeepSeek at its off-peak list price "
                 "(2x in its peak hours), with the repeated system prompt billed at the cache-hit price. "
                 "Auto-decided = the largest share of messages above a confidence threshold whose exact-match "
                 "accuracy is still >= 95 % (Jev: weakest typed answer; LLM: least certain generated token). "
                 "T = sampling temperature. \\* No usable confidence: at T = 0 DeepSeek reports probability 1.0 "
                 "for every generated token, so a threshold cannot separate its answers and it decides all "
                 "messages or none (all only if its accuracy is >= 95 %).")
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(RESULTS / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system"] + [c[0] for c in cols])
        for name, m in metrics.items():
            w.writerow([name] + [m[c[0]] for c in cols])
    return metrics


def plot_threshold(runs, metrics):
    rows = runs["jev"]
    sw = sweep(rows)
    best = best_threshold(rows)
    with open(RESULTS / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sw[0]))
        w.writeheader()
        w.writerows(sw)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ts = [s["threshold"] for s in sw]
    ax.plot(ts, [s["auto_accuracy"] for s in sw], color="#2563eb", lw=2.2, label="accuracy of auto-decided messages")
    ax.plot(ts, [s["auto_share"] for s in sw], color="#16a34a", lw=2.2, label="share decided automatically")
    ax.fill_between(ts, [s["auto_share"] for s in sw], 1, color="#f97316", alpha=0.12, label="share sent to a human")
    ax.axhline(TARGET, color="#6b7280", ls="--", lw=1)
    ax.text(0.01, TARGET + 0.01, "95% target", color="#6b7280", fontsize=9)
    if best:
        ax.axvline(best["threshold"], color="#dc2626", ls=":", lw=1.5)
        ax.annotate(f"t = {best['threshold']:.2f}\n{best['auto_share']:.0%} automatic at "
                    f"{best['auto_accuracy']:.0%} accuracy\n{best['human_share']:.0%} to a human",
                    xy=(best["threshold"], best["auto_share"]), xytext=(0.5, 0.25), fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "#dc2626"}, color="#991b1b")
    ax.set_xlabel("confidence threshold (decision confidence = weakest of the three answers)")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, 1)
    ax.set_title("Jev: accuracy vs confidence threshold (80 test messages)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.3)

    # right: accuracy vs automation for every system; a system without a confidence is one point (decides all)
    colors = iter(["#16a34a", "#9333ea", "#ea580c", "#0891b2"])
    markers = iter(["o", "s", "D", "^", "v"])
    lowest = 1.0
    for name, run in runs.items():
        m = metrics[name]
        color = "#2563eb" if name == "jev" else next(colors)
        if m["confidence_kind"] == "informative":
            xs, ys = zip(*coverage_curve(run))
            lowest = min(lowest, min(y for x, y in zip(xs, ys) if x >= 0.1))
            cov = m["coverage"]
            ax2.plot(xs, ys, color=color, lw=2.2 if name == "jev" else 1.6,
                     label=label(name) + (f": {cov['auto_share']:.0%} auto at ≥ 95 %" if cov else ""))
            if cov:
                ax2.scatter([cov["auto_share"]], [cov["auto_accuracy"]], color=color, s=45, zorder=3)
        else:
            lowest = min(lowest, m["exact_match"])
            why = "confidence always 1.0" if m["confidence_kind"] == "constant" else "no confidence"
            ax2.scatter([1.0], [m["exact_match"]], s=60, marker=next(markers), color=color, zorder=3,
                        label=f"{label(name)}: {why}, decides everything")
    ax2.axhline(TARGET, color="#6b7280", ls="--", lw=1)
    ax2.set_xlabel("share of messages decided automatically (most confident first)")
    ax2.set_ylabel("exact-match accuracy of those messages")
    ax2.set_xlim(0, 1.03)
    ax2.set_ylim(max(0.0, math.floor((lowest - 0.05) * 20) / 20), 1.01)
    ax2.set_title("Accuracy vs automation, every system")
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "jev_threshold.png", dpi=150)
    plt.close(fig)
    return best


def write_tricky(runs, metrics):
    names = list(runs)
    tricky = {name: {r["id"]: r for r in by_set(rows, "tricky")} for name, rows in runs.items()}
    ids = sorted(next(iter(tricky.values())))
    thresholds = {n: metrics[n]["coverage"]["threshold"] for n in names if metrics[n]["coverage"]}
    lines = ["| # | Kind | Message | Gold | " + " | ".join(label(n) for n in names) + " |",
             "|---" * (4 + len(names)) + "|"]
    for i in ids:
        first = tricky[names[0]][i]
        cells = []
        for n in names:
            r = tricky[n][i]
            text = fmt_pred(r["pred"], r["gold"])
            if metrics[n]["confidence_kind"] == "informative":
                route = "auto" if n in thresholds and r["confidence"] >= thresholds[n] else "→ human"
                text += f" (conf {r['confidence']:.2f}, {route})"
            cells.append(text)
        text = first["text"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {i} | {first['kind']} | {text} | {fmt_pred(first['gold'])} | " + " | ".join(cells) + " |")
    no_conf = [label(n) for n in names if metrics[n]["confidence_kind"] != "informative"]
    lines.append("\n**bold✗** = wrong field. conf = the system's confidence (Jev: the weakest of its three answers; "
                 "LLM: its least certain generated token). 'auto' = at or above that system's own threshold for "
                 "≥ 95 % accuracy on the test set (" + ", ".join(f"{label(n)}: t = {t:.3f}" for n, t in thresholds.items())
                 + "), '→ human' = below it."
                 + (f" {', '.join(no_conf)}: no usable confidence (1.0 for every answer), so every answer would be "
                    "used as it is." if no_conf else ""))
    (RESULTS / "tricky_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_workflow(path=RESULTS / "workflow.png"):
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec="#334155", lw=1.3))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, weight="bold")
        ax.text(x + 0.15, y + h - 0.68, body, ha="left", va="top", fontsize=8.6, family="monospace", linespacing=1.45)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "color": "#334155", "lw": 1.4})
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=8.6, color="#334155")

    box(0.1, 2.0, 2.3, 1.6, "Customer message", "ru / kz / en / mix\nany length", "#f1f5f9")
    box(2.9, 0.55, 3.4, 4.5, "Jev 1.14: one request", "state = {customer_message}\n\n"
        "category  CHOICE\n  billing · network · tariff\n  sim · account · other\n\n"
        "urgency   SCORE 0..2\n  0 low · 1 medium · 2 high\n\n"
        "security_incident  NOUL\n  P(fraud / stolen phone /\n    hijacked account / ...)\n\n"
        "pressure  NOUL\n  P(threat to leave, sue,\n    complain / repeat contact)", "#dbeafe")
    box(7.6, 0.95, 3.7, 3.7, "Python code (thresholds)", "category = choice\n"
        "urgency  = most likely level\n  if P(security) >= 0.5:\n     urgency = high\n"
        "escalate = max(P(security),\n           P(pressure)) >= 0.5\n\n"
        "confidence = min(\n  conf(category),\n  conf(urgency),\n  |2·P(escalate) - 1|)", "#fef3c7")
    box(12.0, 3.15, 2.9, 1.9, "Automatic routing", "confidence >= t\n-> team queue,\n   SLA by urgency,\n   senior agent if\n   escalate", "#dcfce7")
    box(12.0, 0.4, 2.9, 1.9, "Human agent", "confidence < t\n-> a person checks\n   the suggestion\n   and decides", "#ffedd5")
    arrow(2.4, 2.8, 2.9, 2.8)
    arrow(6.3, 2.8, 7.6, 2.8)
    ax.text(6.95, 3.0, "typed answers\n+ probabilities", ha="center", va="bottom", fontsize=8.3, color="#334155")
    arrow(11.3, 3.2, 12.0, 4.1, "yes")
    arrow(11.3, 2.4, 12.0, 1.35, "no")
    ax.text(7.5, 5.35, "Altyn Mobile message routing with Jev: typed questions + code + confidence threshold",
            ha="center", fontsize=12, weight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("no raw_*.jsonl in a1_jev/results yet, run a1_jev/run.py first")
    draw_workflow()
    metrics = write_table(runs)
    print((RESULTS / "results_table.md").read_text(encoding="utf-8"))
    if "jev" in runs:
        print("Jev best threshold:", plot_threshold(runs, metrics))
    else:
        print("no Jev results yet: threshold plot skipped")
    for name, m in metrics.items():
        print(f"{name}: auto-decided at >= 95 % accuracy:", m["coverage"])
    write_tricky(runs, metrics)


if __name__ == "__main__":
    matplotlib.use("Agg")   # files only, no windows (the notebook imports this module and keeps its own backend)
    main()
