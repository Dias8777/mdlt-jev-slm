# Dataset card: Altyn Mobile message router (Assignment 2)

Training data for fine-tuning a small language model to route customer messages of *Altyn Mobile*, a fictional
mobile operator in Kazakhstan. Each message gets three labels, returned as one JSON object:

```json
{"category": "sim", "urgency": "high", "escalate": true}
```

| field | values |
|---|---|
| `category` | `billing`, `network`, `tariff`, `sim`, `account`, `other` (which team solves the main problem) |
| `urgency` | `low`, `medium`, `high` |
| `escalate` | `true` / `false` (fraud or a security incident, a threat to leave / sue / complain, repeated contact) |

The full labelling guide is in [`common/task.py`](../common/task.py). The same text is used to generate the data, to
label it, as the system prompt of the model (before and after fine-tuning), and in the LLM baseline of Assignment 1.

## Files (`a2_slm/data/`)

| file | rows | what it is |
|---|---|---|
| `train.jsonl` | 819 | training pairs in chat format |
| `valid.jsonl` | 89 | validation pairs (validation loss only) |
| `test.jsonl` | 80 | the hand-written test set of Assignment 1 (`data/test_tickets.jsonl`) in the same format |
| `tricky.jsonl` | 10 | the 10 tricky inputs of Assignment 1 |
| `sample_20.jsonl`, `sample_20.md` | 20 | a sample of training rows, exactly as trained, and the same rows as a table with how each label was decided |
| `distribution.md` | | label and language counts per split |
| `dataset_full.jsonl` | 908 | train + valid with metadata: language style, tone, length, the generator's intended labels, every labeller's vote, label source, similarity to the test set |
| `raw_generated.jsonl` | 960 | everything the generator wrote, before cleaning |
| `cleaned_with_votes.jsonl` | 942 | after cleaning, with all votes (including the 34 rows dropped later for having no majority) |
| `stats.json` | | the numbers below, written by the pipeline |
| `cache/` | | every API response, so `python -m a2_slm.make_dataset` re-runs for free and gives the same result |

Row format (TRL's conversational format; the assistant turn is the target):

```json
{"messages": [{"role": "system", "content": "You route customer messages for Altyn Mobile … (labelling guide)"},
              {"role": "user", "content": "Украли телефон в автобусе! Срочно заблокируйте мой номер"},
              {"role": "assistant", "content": "{\"category\": \"sim\", \"urgency\": \"high\", \"escalate\": true}"}]}
```

## How it was made

Script: [`make_dataset.py`](make_dataset.py) (needs a DeepSeek API key; total API cost $0.24).

1. **Generate.** 960 messages in 135 requests to `deepseek-flash` (temperature 1.0). The plan fixes how many
   messages each label combination gets (`CELLS`, 160 per category), and every message in a request gets its own
   randomly drawn style: one of 9 language styles, a tone (polite, neutral, irritated, very angry in capitals, sarcastic, a confused elderly customer, in a hurry,
   friendly) and a length (2–6 words to 4–6 sentences).
   The generator gets topic hints per category, the labelling guide and the target labels; it never sees the test set.
2. **Clean** (960 → 942). Unicode / whitespace / quote normalisation; dropped 1 message outside 8–600 characters,
   messages with leaked labels or placeholders (`urgency:`, `[…]`, `XXX`; none found), 2 exact duplicates (after lower-casing and removing
   punctuation), near-duplicates (TF-IDF character 3–5-gram cosine ≥ 0.85: none found) and **15 messages too similar
   to a test or tricky message** (cosine ≥ 0.60). The most similar kept message has cosine 0.599 to its closest test
   message, so the test set is not leaked into training.
3. **Label again, blind** (verification). A different, larger model, `deepseek-v4-pro` (temperature 0), labelled all
   942 messages from the guide alone, 20 per request, without seeing the intended labels.
   Agreement with the generator's intent: **category 92.5 %, urgency 57.1 %, escalate 75.1 %**. The generator often
   wrote a message that, read cold, has a different urgency than it was asked for, so the intended labels alone
   would have been noisy.
4. **Tie-break.** The 544 messages where the two disagreed on at least one field went to a third blind labeller
   (`deepseek-flash` in thinking mode). Every field takes the majority of the three votes (intent, verifier,
   tie-breaker). **34 messages with three different votes on a field were dropped** (33 on urgency: low / medium / high, 1 on
   category).
5. **Split** (908 → 819 / 89). Stratified by category × urgency × escalate, 90 % train, 10 % validation.

Result: 398 rows where the generator and the verifier fully agreed, 510 decided by majority vote; **448 of the 908
kept rows carry labels different from what the generator intended**. The table in `sample_20.md` shows the votes for
20 of them.

## Statistics

| | train | valid | test (hand-written) |
|---|---|---|---|
| category = billing | 152 | 16 | 13 |
| category = network | 151 | 17 | 14 |
| category = tariff | 107 | 11 | 14 |
| category = sim | 135 | 15 | 13 |
| category = account | 138 | 15 | 13 |
| category = other | 136 | 15 | 13 |
| urgency = low | 289 | 31 | 34 |
| urgency = medium | 350 | 39 | 31 |
| urgency = high | 180 | 19 | 15 |
| escalate = true | 160 | 17 | 15 |
| escalate = false | 659 | 72 | 65 |
| **total** | **819** | **89** | **80** |

| language style (train + valid) | rows |
|---|---|
| Russian | 302 |
| Russian, casual | 167 |
| Kazakh–Russian mix | 103 |
| Russian with typos | 84 |
| Kazakh | 80 |
| Kazakh / Russian in Latin letters | 57 |
| Russian with emoji | 44 |
| English | 39 |
| Russian, formal | 32 |

Tokens per training example with the Qwen3 chat template: about 580 on average (≈ 500 of them the system prompt,
≈ 20 the answer).

## The test set

`data/test_tickets.jsonl`: 80 messages written and labelled by hand for Assignment 1, before any training data
existed (source with labels: `data/make_test_sets.py`). Before any model was run, every label was re-checked against
the written guide and 5 labels that contradicted it were fixed, 4 in the test set and 1 in the tricky set (marked
`# audit:` in that file).
Jev, both DeepSeek baselines, the base SLM and the fine-tuned SLM are all scored on exactly these 80 messages.

## Limitations

* **Synthetic messages.** Real customers write shorter, with more typos, voice-to-text errors and screenshots; an
  LLM writes cleaner, more "typical" complaints. Accuracy on real traffic is likely lower than on this data.
* **LLM labels.** Training labels are majority votes of three LLM passes, not human labels. Urgency is the noisiest
  field (57 % first-pass agreement), and the borders low / medium are partly a matter of taste.
* **Small test set.** With 80 messages one message is 1.25 points, and the 95 % confidence interval of an accuracy
  around 80 % is roughly ±9 points, so small differences between systems are not significant.
* **One labelling guide.** Generation, labelling and evaluation all follow the same guide, which keeps the labels
  consistent but also means the data reflects how DeepSeek reads that guide.
