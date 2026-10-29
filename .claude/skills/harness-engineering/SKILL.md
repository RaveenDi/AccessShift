---
name: harness-engineering
title: Harness Engineering for AI Agents
description: Design the scaffolding around an LLM — the tool set, permission model, context-management hooks, verification loops, and orchestration — rather than just the prompt. Use when building or hardening an agentic coding tool, CLI agent, autonomous pipeline, or any system where the harness (not the model) decides what the agent can see, do, and recover from. Covers tool design, sandboxing, approval gates, retries/escalation, observability, and eval-driven iteration.
license: MIT
category: workflow-agents
---

# Harness Engineering for AI Agents

A **harness** is everything around the model: the tools it can call, what context it's fed each
turn, what it's allowed to do unsupervised, how failures are caught and retried, and how progress
is shown to a human. Two agents running the *same model* with the *same prompt* behave completely
differently if one has a good harness and one doesn't. This skill is about designing that
scaffolding deliberately instead of letting it accrete.

## 0. Harness vs. prompt engineering

Prompt engineering optimizes *what you say to the model*. Harness engineering optimizes
*everything else*: what the model can act on, what it's shown, what's verified before and after
each action, and what happens when it's wrong. A weak model in a strong harness often beats a
strong model in a weak harness — the harness is where reliability actually comes from.

## 1. Tool design

- **Small, composable, verifiable tools** beat one giant "do anything" tool. Each tool should do
  one thing, return a result the model can check, and fail loudly with an actionable message —
  not a silent no-op or a stack trace dump.
- **Shape tools around the task's real primitives**, not around the underlying API. If the task is
  "edit a file," don't make the model regenerate the whole file through a raw write call — give it
  a diff/patch tool so mistakes are small and reviewable.
- **Make illegal states unrepresentable.** If a tool call can't possibly succeed given current
  state (e.g. editing a file that was never read), reject it before it reaches the model's
  conversation, with a message that teaches the right sequence.
- **Return just enough, not everything.** A tool that dumps 50k tokens of logs burns context the
  model needed for reasoning. Summarize, paginate, or let the model ask for more.
- **Idempotency and dry-run matter more than cleverness.** An agent will call a tool twice because
  it lost track of state; design for that instead of hoping it won't happen.

## 2. The control loop

Most agent harnesses are a loop: *gather context → model proposes an action → harness validates →
harness executes (or asks a human) → harness feeds back the result → repeat*. Decide, explicitly,
for each step:

- **What's injected into context on every turn** (system state, file tree, recent errors) vs.
  **what's fetched on demand** (a file's contents, a search result). Over-injecting wastes budget
  and buries the model in noise; under-injecting causes it to guess or thrash. See
  `context-engineering`.
- **Where verification happens.** Cheap, deterministic checks (schema validation, linters, type
  checkers, tests) belong in the harness, run automatically after every change — not left to the
  model to remember to run. The harness should make the *right* thing the *easy* thing.
- **How errors re-enter the loop.** A raw exception is useless; the harness should translate it
  into a message that tells the model what happened and what to try differently, so one failure
  doesn't become five identical failures.

## 3. Autonomy, permissions, and sandboxing

- **Tier actions by blast radius and reversibility**, not by how "advanced" they sound. Reading a
  file is free. Editing a file in a repo under version control is cheap to undo. Running `rm -rf`,
  pushing to a shared branch, calling a paid API, or sending a message to another human are not —
  gate those behind explicit approval or a narrower, pre-approved action.
- **Sandbox by default.** Give the agent a disposable environment (container, worktree, VM, scoped
  cloud credentials) wide enough to actually get work done, and treat anything touching production,
  real money, or real external systems as requiring a human in the loop until the harness has
  enough evidence (evals, track record) to trust it unsupervised.
- **Default-deny, allowlist up**, not default-allow with a denylist of things you thought of.
  Denylists always miss the case nobody imagined.
- **Make the permission boundary visible to the model**, not just enforced silently — tell it which
  of its proposed actions needs approval and why, so it plans around that instead of being
  surprised.

## 4. Recovery, retries, and escalation

- **Don't retry blindly.** Retrying the exact same failed action produces the exact same failure.
  Feed back *why* it failed and let the model adjust, or cap retries and escalate to a human/larger
  model after N attempts.
- **Detect loops.** If the agent repeats a near-identical action 3+ times without progress, that's
  a harness signal to interrupt, not a prompt problem to word around.
- **Checkpoint and resume.** Long-running agent work should be resumable from the last good state,
  not restarted from scratch after a crash, timeout, or context overflow.
- **Graceful degradation beats a hard stop.** When a tool is unavailable or a step fails, prefer a
  harness that narrows scope or asks a clarifying question over one that silently hallucinates a
  result or dies with no explanation.

## 5. Observability

- Log every tool call, its input, its result, and whether it was approved/denied/auto-run — this is
  what makes failures debuggable after the fact and what eval data gets built from.
- Surface progress to the human in terms of *outcomes* ("ran tests, 2 failing" ), not raw
  tool-call noise — a harness that only shows chatter trains users to stop reading it.
- Track cost and latency per step; a harness that can't tell you where tokens and time went can't
  be optimized, only guessed at.

## 6. Iterate with evals, not vibes

Build a small, versioned set of representative tasks (including the ones that broke production
last time) and run the harness against them whenever a tool, prompt, or permission rule changes.
A harness change that "feels better" but regresses the eval set is a regression, full stop.
Prefer measuring end-to-end task success over any individual step's output quality — the harness
exists to make the whole loop reliable, not any one call inside it.

## Pairs well with

`context-engineering` (what the loop feeds the model each turn), `ai-engineering` (evals,
observability, and production practices once the harness ships), `skill-loadout-router`
(choosing which skills a harness should load), and testing/security-review skills for the
verification layer.
