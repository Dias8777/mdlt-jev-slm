"""Small client for OpenAI-compatible chat APIs (DeepSeek directly, or any model through Vercel AI Gateway)."""
import math
import time

import requests

from .keys import get_key

PROVIDERS = {
    # DeepSeek's own API. "thinking: disabled" turns off the reasoning mode, so the model answers directly.
    "deepseek": {"base": "https://api.deepseek.com", "key": "DEEPSEEK_API_KEY",
                 "extra": {"thinking": {"type": "disabled"}}},
    # OpenRouter: one key for GPT, Claude, Gemini, ... (model ids like "openai/gpt-5-mini").
    # It returns the exact cost of every call in usage.cost.
    "openrouter": {"base": "https://openrouter.ai/api/v1", "key": "OPENROUTER_API_KEY",
                   "extra": {"usage": {"include": True}}},
    # Vercel AI Gateway: one key for Claude, GPT, Gemini, ... (model ids like "anthropic/claude-haiku-4.5")
    "vercel": {"base": "https://ai-gateway.vercel.sh/v1", "key": "AI_GATEWAY_API_KEY", "extra": {}},
}

# USD per 1M tokens: (input, cached input, output). DeepSeek list prices checked on 2026-11-11 (off-peak;
# DeepSeek charges 2x during its peak hours). DeepSeek caches the repeated system prompt automatically,
# so most input tokens after the first request are billed at the cached price.
PRICES = {
    "deepseek-flash": (0.15, 0.003, 0.60),
    "deepseek-v4-pro": (0.66, 0.022, 1.98),
}


def cost_usd(model, usage):
    """Cost of one call: the provider's own number if it reports one, otherwise tokens x list price."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    if model not in PRICES:
        return float("nan")
    p_miss, p_hit, p_out = PRICES[model]
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    return (miss * p_miss + hit * p_hit + usage.get("completion_tokens", 0) * p_out) / 1e6


def chat(messages, model, provider="deepseek", temperature=0.0, max_tokens=200, json_mode=True, logprobs=False,
         extra=None, timeout=90, retries=4, session=None):
    """One chat completion. Returns dict(text, latency_s, usage, model, finish_reason, token_probs).

    logprobs=True asks for the probability of every generated token (token_probs, None if the API sends none).
    Retries on 429/5xx and timeouts."""
    p = PROVIDERS[provider]
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if logprobs:
        body["logprobs"] = True
    body.update(p["extra"])
    if extra:
        body.update(extra)
    headers = {"Authorization": f"Bearer {get_key(p['key'])}", "Content-Type": "application/json"}
    http = session or requests
    last = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = http.post(f"{p['base']}/chat/completions", json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = repr(e)
            time.sleep(2 ** attempt)
            continue
        dt = time.perf_counter() - t0
        if r.status_code == 200:
            data = r.json()
            choice = data["choices"][0]
            lp = (choice.get("logprobs") or {}).get("content")
            return {"text": choice["message"].get("content") or "", "latency_s": dt, "usage": data.get("usage", {}),
                    "model": data.get("model", model), "finish_reason": choice.get("finish_reason"),
                    "token_probs": [math.exp(t["logprob"]) for t in lp] if lp else None}
        last = f"HTTP {r.status_code}: {r.text[:300]}"
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt + 1)
            continue
        break
    raise RuntimeError(f"{provider}/{model} failed: {last}")
