"""Write the 20-row dataset sample and the numbers for the dataset card.

    python -m a2_slm.dataset_report

-> a2_slm/data/sample_20.jsonl   20 rows exactly as used for training ({"messages": [system, user, assistant]})
-> a2_slm/data/sample_20.md      the same 20 rows as a readable table, with how each label was decided
-> a2_slm/data/distribution.md   label and language counts per split
"""
import json
import random
from collections import Counter

from common.task import CATEGORIES, URGENCY, load_jsonl, save_jsonl
from a2_slm.slm import DATA, gold_of

LANG_NAMES = {"ru": "Russian", "ru_casual": "Russian, casual", "ru_typos": "Russian with typos",
              "kz_ru": "Kazakh-Russian mix", "kz": "Kazakh", "latin": "Kazakh/Russian in Latin letters",
              "en": "English", "formal": "Russian, formal", "emoji": "Russian with emoji"}


def pick_sample(rows, n=20, seed=7):
    """Deterministic, varied sample: every category, every urgency, both escalate values, many languages,
    and some rows whose label was changed during cleaning."""
    rng = random.Random(seed)
    rows = [r for r in rows if r["split"] == "train"]
    rng.shuffle(rows)
    chosen, seen_lang = [], Counter()
    want = {c: 0 for c in CATEGORIES}
    # first pass: 3 per category, prefer new languages, keep both relabelled and agreed rows
    for r in rows:
        c = r["category"]
        if want[c] >= 3 or seen_lang[r["lang"]] >= 3:
            continue
        want[c] += 1
        seen_lang[r["lang"]] += 1
        chosen.append(r)
    # second pass: top up with escalate=true and relabelled rows until n
    for r in rows:
        if len(chosen) >= n:
            break
        if r in chosen:
            continue
        esc = sum(x["escalate"] for x in chosen)
        rel = sum(x["relabelled"] for x in chosen)
        if (r["escalate"] and esc < 5) or (r["relabelled"] and rel < 7):
            chosen.append(r)
    for r in rows:
        if len(chosen) >= n:
            break
        if r not in chosen:
            chosen.append(r)
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return sorted(chosen[:n], key=lambda r: (order[r["category"]], URGENCY.index(r["urgency"])))


def short(labels):
    return f"{labels['category']}/{labels['urgency']}/{'esc' if labels['escalate'] else 'no'}"


def main():
    full = load_jsonl(DATA / "dataset_full.jsonl")
    train_by_text = {r["messages"][1]["content"]: r for r in load_jsonl(DATA / "train.jsonl")}
    sample = pick_sample(full)
    save_jsonl([train_by_text[r["text"]] for r in sample], DATA / "sample_20.jsonl")

    lines = ["| # | Language | Message | Answer the model learns | How the label was decided |", "|---|---|---|---|---|"]
    for i, r in enumerate(sample, 1):
        answer = train_by_text[r["text"]]["messages"][2]["content"]
        if r["label_source"] == "agreed":
            how = "generator intent = blind verifier"
        else:
            votes = [f"intent {short(r['intended'])}", f"verifier {short(r['verifier'])}"]
            if r.get("tiebreaker"):
                votes.append(f"tie-breaker {short(r['tiebreaker'])}")
            how = "majority vote: " + "; ".join(votes) + (" (relabelled)" if r["relabelled"] else "")
        text = r["text"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {i} | {LANG_NAMES.get(r['lang'], r['lang'])} | {text} | `{answer}` | {how} |")
    (DATA / "sample_20.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # distribution tables
    splits = {"train": [r for r in full if r["split"] == "train"], "valid": [r for r in full if r["split"] == "valid"],
              "test (hand-written)": [gold_of(r) for r in load_jsonl(DATA / "test.jsonl")]}
    out = ["| | " + " | ".join(splits) + " |", "|---" * (len(splits) + 1) + "|"]
    for field, values in (("category", CATEGORIES), ("urgency", URGENCY), ("escalate", [True, False])):
        for v in values:
            out.append(f"| {field} = {str(v).lower()} | " + " | ".join(str(sum(r[field] == v for r in rows))
                                                                    for rows in splits.values()) + " |")
    out.append("| **total** | " + " | ".join(str(len(rows)) for rows in splits.values()) + " |")
    langs = Counter(r["lang"] for r in full)
    out += ["", "| Language style (train + valid) | rows |", "|---|---|"]
    out += [f"| {LANG_NAMES.get(k, k)} | {v} |" for k, v in langs.most_common()]
    (DATA / "distribution.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]), "\n...\n", "\n".join(out), sep="")
    print("sample: escalate", sum(r["escalate"] for r in sample), "relabelled", sum(r["relabelled"] for r in sample),
          "langs", Counter(r["lang"] for r in sample))


if __name__ == "__main__":
    main()
