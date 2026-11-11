## System One vs LLM

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
