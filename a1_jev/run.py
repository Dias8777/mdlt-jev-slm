"""Run one system over the 80 hand-labelled test messages and the 10 tricky ones.

    python -m a1_jev.run jev                                      # Jev (OPENROUTER_, TYPESAFE_ or AI_GATEWAY_API_KEY)
    python -m a1_jev.run llm deepseek-flash                       # LLM baseline through the DeepSeek API
    python -m a1_jev.run llm deepseek-v4-pro
    python -m a1_jev.run llm deepseek-flash --temperature 1 --name deepseek-flash-t1   # informative token probabilities
    python -m a1_jev.run llm openai/gpt-5-mini --provider openrouter

Every answer is appended to a1_jev/results/raw_<name>.jsonl as soon as it arrives, with latency, tokens and
cost. Messages already in that file are skipped, so a crash or a re-run never pays twice. Requests go one at
a time, so the measured latency is the latency of a single decision.
"""
import argparse
import json
import time
from pathlib import Path

from a1_jev import workflow
from common import llm
from common.task import FIELDS, TEST_PATH, TRICKY_PATH, chat_messages, load_jsonl, parse_answer

RESULTS = Path(__file__).resolve().parent / "results"


def items():
    out = []
    for name, path in (("test", TEST_PATH), ("tricky", TRICKY_PATH)):
        for r in load_jsonl(path):
            out.append({"set": name, "id": r["id"], "kind": r.get("kind"), "text": r["text"],
                        "gold": {f: r[f] for f in FIELDS}})
    return out


def run_jev(item, args):
    res = workflow.ask_jev(item["text"], backend=args.backend, model=args.model)
    d = workflow.decide(res["answers"])
    return {"pred": {f: d[f] for f in FIELDS}, "status": "ok", "confidence": d["confidence"], "decision": d,
            "raw": res["answers"], "latency_s": res["latency_s"], "usage": res["usage"], "cost": res["cost"],
            "model": res["model"], "backend": res["backend"]}


def run_llm(item, args):
    res = llm.chat(chat_messages(item["text"]), model=args.model, provider=args.provider, max_tokens=100,
                   temperature=args.temperature, logprobs=not args.no_logprobs)
    pred, status = parse_answer(res["text"])
    probs = res["token_probs"]
    # confidence of an LLM answer = the probability of its least certain generated token (None without logprobs)
    return {"pred": pred, "status": status, "raw": res["text"], "latency_s": res["latency_s"],
            "usage": res["usage"], "cost": llm.cost_usd(args.model, res["usage"]), "model": res["model"],
            "finish_reason": res["finish_reason"], "temperature": args.temperature,
            "confidence": min(probs) if probs else None,
            "token_probs": [round(x, 6) for x in probs] if probs else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=["jev", "llm"])
    ap.add_argument("model", nargs="?", help="LLM model id (required for llm); Jev version for jev")
    ap.add_argument("--provider", default="deepseek", help="llm: deepseek | openrouter | vercel")
    ap.add_argument("--backend", default=None, help="jev: openrouter | typesafe | vercel (default: the first key that is set)")
    ap.add_argument("--name", help="output name (default: jev or the model name)")
    ap.add_argument("--no-logprobs", action="store_true", help="llm: do not ask for token probabilities")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="llm: sampling temperature (default 0). DeepSeek reports token probabilities after "
                         "temperature scaling, so at 0 every token has probability 1.0; use 1 to see the model's own")
    args = ap.parse_args()
    if args.system == "llm" and not args.model:
        ap.error("llm needs a model id")
    name = args.name or ("jev" if args.system == "jev" else args.model.split("/")[-1])
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"raw_{name}.jsonl"
    done = {(r["set"], r["id"]) for r in load_jsonl(out)} if out.exists() else set()
    todo = [it for it in items() if (it["set"], it["id"]) not in done]
    print(f"{name}: {len(done)} already done, {len(todo)} to run -> {out.name}")
    fn = run_jev if args.system == "jev" else run_llm
    with open(out, "a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            r = fn(it, args)
            row = {"system": name, **it, **r, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            mark = "ok " if r["pred"] == it["gold"] else "ERR"
            print(f"[{i}/{len(todo)}] {it['set']}#{it['id']:<2} {mark} {r['latency_s']:.2f}s "
                  f"pred={r['pred']} gold={it['gold']}", flush=True)


if __name__ == "__main__":
    main()
