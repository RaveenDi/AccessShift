---
name: ai-engineering
title: AI Engineering (Shipping LLM Features to Production)
description: Take an LLM-powered feature from prototype to a reliable, observable, cost-aware production system — evals, prompt/version management, retrieval vs. fine-tuning vs. few-shot decisions, structured outputs, guardrails, latency/cost tradeoffs, and rollout strategy. Use when a demo needs to become a shipped feature, when "it works on my examples" isn't good enough, or when someone asks how to test, monitor, or scale an LLM integration.
license: MIT
category: ai-ml
---

# AI Engineering

AI engineering is the gap between "the model did the right thing in my test prompt" and "this
feature works for real users, at scale, reliably, and we'd know within minutes if it broke."
It borrows from software engineering (testing, versioning, observability, rollout) and applies it
to a component that is probabilistic and prompt-sensitive instead of deterministic.

## 1. Evals before features

- **Write the eval set before you tune the prompt.** A representative set of real (or
  realistic) inputs with a way to score the output — exact match, rubric-graded, LLM-as-judge, or
  a human review queue — is the only way to know if a change helped or hurt.
- **Capture failures from production into the eval set.** Every bug report is a missing test
  case; the eval set should grow every time something breaks.
- **Score on the actual task outcome**, not surface similarity to a reference answer. If the task
  is "extract these 5 fields," score field-by-field accuracy, not BLEU-style text overlap.
- **Use an LLM judge carefully**: give it a rubric, few-shot examples of each score, and
  spot-check its judgments against humans regularly — judges drift and can be gamed by verbose
  but wrong answers.

## 2. Choosing how knowledge gets to the model

Pick the cheapest technique that clears the eval bar — don't reach for fine-tuning by default:

| Need | Reach for |
|---|---|
| Teach a response *format* or *style* | Few-shot examples in the prompt |
| Give access to information that changes often or is too large to fit in context | Retrieval (RAG): chunk, embed, retrieve top-k, cite sources |
| Teach a narrow, stable skill cheaper/faster at high volume | Fine-tuning (only after prompting + RAG hit a ceiling) |
| Combine external facts with live computation | Tool calling / function calling |
| Need the model to act over multiple steps | Agent loop — see `harness-engineering` |

RAG quality depends far more on **retrieval** (chunking strategy, hybrid search, reranking) than
on the generation prompt — if answers are wrong or ungrounded, instrument retrieval first.

## 3. Structured outputs and guardrails

- Prefer **constrained/structured output** (JSON schema, function-calling, grammars) over asking
  the model to "please return valid JSON" and hoping — it eliminates a whole class of parsing
  failures.
- **Validate, don't trust.** Schema-validate every structured response before it reaches
  downstream code; treat a validation failure as a retryable error, not a crash.
- **Guard against the specific failure modes of your task**: hallucinated citations, unsafe
  content, prompt injection from untrusted input (web pages, documents, tool output), and
  over-confident wrong answers on out-of-scope questions. Add an explicit refusal/"I don't know"
  path — a model that always answers is more dangerous than one that sometimes declines.
- **Treat anything an LLM reads from outside the user's own message as untrusted data**, not
  instructions — this is the same boundary a harness enforces for tool output.

## 4. Latency, cost, and model selection

- **Right-size the model per step.** Route cheap/fast classification or extraction to a small
  model; reserve the largest model for the step that actually needs its reasoning. Multi-model
  pipelines are usually both cheaper and faster than one big model doing everything.
- **Cache aggressively**: prompt caching for repeated system/context prefixes, response caching
  for repeated or near-duplicate queries, embeddings caching for unchanged documents.
- **Stream output** wherever a human is waiting on it — perceived latency matters as much as
  actual latency.
- **Budget tokens like a resource**, not an afterthought: track cost per request and per feature
  in production, and alert on cost/latency regressions the same way you'd alert on error rate.

## 5. Versioning and rollout

- **Version prompts and model IDs like code** — reviewed diffs, not edited in place in a config
  the model happens to read. A prompt change is a behavior change and deserves the same scrutiny.
- **Pin model versions deliberately.** "Always use the latest" silently changes behavior
  underneath you; upgrade on purpose, against the eval set, and roll forward with a changelog.
- **Canary and A/B** prompt/model changes against real traffic before a full rollout, with the
  eval metric as the promotion gate, not just "looks good in a few manual checks."
- **Keep a fallback path**: a cheaper/older model or a cached/static response for when the
  primary model or provider is degraded or rate-limited.

## 6. Observability in production

Log (with privacy/PII handling in mind): the prompt version, model, input, output, token counts,
latency, cost, and eval/guardrail outcomes for every call. This is what lets you debug a user
complaint, catch a silent regression after a provider's model update, and build the next eval set.
Monitor eval-set performance continuously where possible (shadow evals on live traffic), not just
at deploy time.

## Pairs well with

`context-engineering` (what goes into the prompt/retrieval that feeds the model),
`harness-engineering` (the control loop and permissions for multi-step/agentic features),
testing-strategy skills (adapting eval practice to CI), and security-review skills (prompt
injection and data handling).
