"""Assignment 1: the Jev workflow for routing Altyn Mobile customer messages.

Jev (TypeSafe's first "System One" model) does not write text. It answers typed questions about the
content in `state` with probabilities. One request per message asks four questions:

    category          choice  6 teams (definitions from common/task.py)
    urgency           score   3 ordered levels: 0 = low, 1 = medium, 2 = high
    security_incident noul    P(fraud or a security incident)
    pressure          noul    P(threat to leave / sue / complain, or "I already wrote several times")

Plain Python (decide() below) turns the four typed answers into the three decisions the business needs
(category, urgency, escalate) plus one confidence number. Messages whose confidence is below a threshold
go to a human instead of being routed automatically.

Access: Jev is called through OpenRouter (OPENROUTER_API_KEY), TypeSafe directly (TYPESAFE_API_KEY) or
Vercel AI Gateway (AI_GATEWAY_API_KEY, its TypeSafe-compatible route). All three serve the same System One API,
so only the URL, the key and the model id differ. Keys come from .env or the environment (common/keys.py) and
never appear in code or notebooks.
"""
import time

import requests

from common.keys import get_key
from common.task import CATEGORY_RULES, URGENCY, URGENCY_RULES

JEV_MODEL = "jev-1.14"          # pinned version (Oct 2026) so the results are reproducible
JEV_PRICE_PER_M_INPUT = 0.042   # USD per 1M input tokens; output tokens are free
BACKENDS = {   # tried in this order; the first one whose key is set is used
    "openrouter": {"url": "https://openrouter.ai/api/v1/systemone", "key": "OPENROUTER_API_KEY", "model": JEV_MODEL},
    "typesafe": {"url": "https://api.typesafe.ai/v1/systemone", "key": "TYPESAFE_API_KEY", "model": JEV_MODEL},
    "vercel": {"url": "https://ai-gateway.vercel.sh/typesafe/v1/systemone", "key": "AI_GATEWAY_API_KEY",
               "model": "typesafe-ai/jev-1.14.0"},
}

# ---------------------------------------------------------------- the four typed questions
QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which team of the mobile operator must solve the customer's main problem? "
                        "If the message has two problems, take the first one.",
        "criteria": CATEGORY_RULES,
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the customer's message?",
        "criteria": [URGENCY_RULES[u] for u in URGENCY],   # level 0 = low, 1 = medium, 2 = high
    },
    "security_incident": {
        "type": "noul",
        "instructions": "Does the customer report fraud or a security incident?",
        "criteria": {
            "true": "a lost or stolen phone, a hijacked account, a SIM card re-issued without the customer's "
                    "consent, a data leak, or money stolen through the customer's number",
            "false": "anything else, including an unexplained or wrong charge, an unwanted subscription, "
                     "anger or sarcasm",
        },
    },
    "pressure": {
        "type": "noul",
        "instructions": "Does the customer threaten to escalate, or say they have already contacted support "
                        "several times about this problem?",
        "criteria": {
            "true": "threatens to leave for another operator, to sue, or to complain to a regulator, a ministry "
                    "or social media; or says this is already their second, third or later request",
            "false": "no such threat and no repeated contacts; anger, sarcasm or a single complaint alone "
                     "do not count",
        },
    },
}

# ---------------------------------------------------------------- thresholds used by the code
TAU_ESCALATE = 0.5   # escalate if P(security_incident) or P(pressure) >= this
TAU_SECURITY = 0.5   # labelling guide: a lost/stolen phone, hijacked account or SIM, stolen money are "high"


def noul_confidence(p):
    """TypeSafe's recommended confidence for a yes/no probability: distance from 0.5, scaled to 0..1."""
    return abs(2 * p - 1)


def decide(answers, tau_escalate=TAU_ESCALATE, tau_security=TAU_SECURITY):
    """Combine Jev's typed answers into the final decision.

    category = the choice Jev picked; confidence = Jev's choice confidence
    urgency  = the most likely score level; if P(security_incident) >= tau_security the rule from the
               labelling guide forces "high" (confidence then comes from the security answer)
    escalate = max(P(security_incident), P(pressure)) >= tau_escalate
    confidence of the whole decision = the weakest of the three (a chain is as strong as its weakest link)
    """
    cat = answers["category"]
    category, c_cat = cat["choice"], float(cat["confidence"])

    urg = answers["urgency"]
    probs = {int(k): float(v) for k, v in urg["probabilities"].items()}
    level = max(probs, key=probs.get)
    urgency, c_urg = URGENCY[level], float(urg["confidence"])

    p_sec = float(answers["security_incident"]["noul"])
    p_pre = float(answers["pressure"]["noul"])
    rule = None
    if p_sec >= tau_security and urgency != "high":
        urgency, c_urg, rule = "high", noul_confidence(p_sec), "security => high"

    p_esc = max(p_sec, p_pre)
    escalate = p_esc >= tau_escalate
    c_esc = noul_confidence(p_esc)

    return {
        "category": category, "urgency": urgency, "escalate": escalate,
        "confidence": min(c_cat, c_urg, c_esc),
        "conf_category": c_cat, "conf_urgency": c_urg, "conf_escalate": c_esc,
        "urgency_score": float(urg["score"]), "p_security": p_sec, "p_pressure": p_pre, "rule": rule,
    }


def default_backend():
    for name, b in BACKENDS.items():
        if get_key(b["key"], required=False):
            return name
    raise RuntimeError("No Jev key. Put OPENROUTER_API_KEY, TYPESAFE_API_KEY or AI_GATEWAY_API_KEY into .env, "
                       "see .env.example")


def ask_jev(text, backend=None, model=None, session=None, timeout=60, retries=5):
    """One Jev request with all four questions. Returns answers, usage, cost (USD), latency and model."""
    backend = backend or default_backend()
    b = BACKENDS[backend]
    model = model or b["model"]
    body = {"model": model, "state": {"customer_message": text}, "questions": QUESTIONS}
    headers = {"Authorization": f"Bearer {get_key(b['key'])}", "Content-Type": "application/json"}
    http = session or requests
    last = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = http.post(b["url"], json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(2 ** attempt)
            continue
        latency = time.perf_counter() - t0
        if r.status_code == 200:
            data = r.json()
            usage = data.get("usage", {})
            cost = usage.get("cost")   # OpenRouter reports the exact charge; TypeSafe reports tokens only
            if cost is None:
                cost = usage.get("input_tokens", 0) * JEV_PRICE_PER_M_INPUT / 1e6
            return {"answers": data["answers"], "usage": usage, "cost": float(cost), "latency_s": latency,
                    "model": data.get("model", model), "backend": backend}
        last = f"HTTP {r.status_code}: {r.text[:300]}"
        if r.status_code in (429, 500, 502, 503, 504, 524, 529):
            wait = float(r.headers.get("Retry-After") or 0) or 2 ** attempt + 1
            time.sleep(min(wait, 30))
            continue
        break
    raise RuntimeError(f"Jev via {backend} failed: {last}")


if __name__ == "__main__":
    import json
    import sys

    text = " ".join(sys.argv[1:]) or "Украли телефон в автобусе! Срочно заблокируйте мой номер"
    res = ask_jev(text)
    print(json.dumps(res["answers"], ensure_ascii=False, indent=1))
    print(decide(res["answers"]))
    print(f"{res['latency_s']:.2f}s  ${res['cost']:.8f}  {res['usage']}  {res['model']} via {res['backend']}")
