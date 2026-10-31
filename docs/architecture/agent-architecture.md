# Agent architecture

Checked against the code on 2026-10-29 (`apps/api/app/{pipeline,agent/*,providers/*,telemetry,config}.py`, `evals/`).
Scope: a calibrated **tabletop model**. Not accessibility compliance, not a real-world measurement.
Principle: **models perceive and interpret; deterministic code decides connectivity.**

## Hierarchy (one pipeline, human in the loop)
`Perceiver -> Calibrator -> Planner Agent -> Human confirm -> (person moves objects) -> Perceiver -> Calibrator -> Verifier -> Recovery turn (at most once)`

| Agent | Kind | Meta model (`meta` / `replay` mode) | `mock` mode (default without a key) |
|---|---|---|---|
| Perceiver | model | `muse-spark-1.3`: named object boxes (tag text, lowercased), 4 board corners, check reference, all on a 0-1000 normalized scale that code converts to pixels. `sam-3.1`: one box-prompted call for all objects plus the board and the check ruler (ids after the objects; max 16 per call; reply format UNVERIFIED until the next live run). Board corners and ruler end points are then read from those two masks (`domain/refine.py`); Spark's values are only seeds and the fallback, and the trace records `refined` or `seed` for each | OpenCV colour segmentation on the synthetic image |
| Calibrator | deterministic | none | same code |
| Planner Agent | model + tools | `muse-spark-1.3` with tool calling; reads the constraint text, picks tools | scripted policy (`mock_policy.py`) that drives the same loop and tools |
| Human confirm | person | none | API call |
| Verifier | deterministic | none (consumes a fresh Perceiver result) | same code |
| Recovery turn | Planner Agent, phase `recovery` | `muse-spark-1.3`, own prompt `pr-1` | same scripted policy |

The old standalone Constraint Interpreter (`providers.*.interpret`) still exists but the pipeline no longer calls it; the Planner Agent interprets the text. Mode: `MODEL_PROVIDER=mock|meta|replay`, default `meta` if `MODEL_API_KEY` is set, else `mock`.
Run states: measuring -> needs_input | needs_confirmation -> planning -> awaiting_recapture -> verifying -> ready | needs_remeasurement | failed.

## Bounded tool-calling planner (`agent/loop.py`, `agent/tools.py`)
- Tools, all server-side and deterministic: `get_layout`, `check_route(clearance_cm, start, end)`, `search_moves(max_moves, locked_ids)`, `explain_move(move_id)`, `ask_user(question)`, `finish(plan_id, summary)`. Strict argument whitelist; clearance must be 20-300 cm; unknown ids, anchors or tools return a tool error the model can recover from.
- Caps (`config.AgentCaps`, env-overridable): 8 tool calls and 8 model turns per plan phase; 6 and 6 in recovery; `ask_user` at most 2 per run; `max_moves` 1-3; one repair attempt for a malformed or tool-less reply.
- `ask_user` pauses the run (`needs_input`). `POST /api/runs/{id}/answer` appends the answer (500 chars, marked as data) and resumes the same conversation. It is disabled in recovery.
- Recovery: if the Verifier returns `route_still_blocked` or `route_borderline`, the agent gets the **observed** layout, the verdict and the problems, keeps all locks, and runs once (`MAX_RECOVERIES = 1`). A corrective plan with moves becomes the new baseline and waits for another recapture. `route_open_but_deviates_from_plan` and `needs_remeasurement` get no recovery turn.
- A cap hit, repeated malformed output or an unavailable model ends as `unresolved`: the pipeline asks the person for `clearance_cm`, `start_name`, `end_name` and falls back to the deterministic search. The last search result is exposed as `partial_plan_id` and never promoted to a final plan.

## Deterministic boundaries
- Only `check_route` and `search_moves` results carry route status. `finish` accepts only a `plan_id` that `search_moves` produced; the free-text `summary` is shown as `note_is_unverified`. No code path lets a model set route status.
- Pixels become centimetres only through the homography from 4 board corners. An independent known length must agree within 3% or calibration is rejected (also: collinear or oblique corners, strong perspective, board under 10% of the image).
- Connectivity: obstacles expanded by clearance/2 plus an uncertainty margin, 4-connected path on a 5 cm raster. Status is `open`, `blocked` or `borderline`; `borderline` is never accepted as clear.
- Search: finite grid (default 10 cm) of same-orientation placements, locks respected (route endpoint objects are always locked), 0 to `max_moves` moves, minimum total displacement. "Minimal" is claimed only within the enumerated set and only if the search finished; statuses `already_open | plan_found | plan_found_search_incomplete | no_solution_in_set | search_incomplete`. Complete results are cached by input hash; time-truncated ones are not.
- Verifier: the recapture is perceived and calibrated independently (the plan never supplies positions). Per object `moved_as_planned | unchanged | not_moved | moved_to_wrong_place | unexpected_move`, then the route is recomputed from observed geometry. Verdicts: `verified_open`, `route_open_but_deviates_from_plan`, `route_still_blocked`, `route_borderline`, `needs_remeasurement`.
- The person confirms footprints and locks (human locks override the agent's) and must accept low-fit footprints before the agent plans or the plan is accepted.

## Record and replay (`providers/recording.py`)
- `meta`: live, billed; every successful request/response pair is saved under `RECORDINGS_DIR` (gitignored) keyed by sha256 of the normalized request (long strings such as images are hashed). Errors are not recorded; the Authorization header is never read into the key or the file.
- `replay`: same adapters, answers only from recordings, no key and no network. A miss raises `ReplayMissError` (not a `ProviderError`, so degradation paths cannot swallow it). Replay works for the multi-turn loop because tool results contain no clocks or random ids.
- `mock` is plumbing only and never reported as model results.

## Telemetry (`telemetry.py`, `GET /api/runs/{id}/trace`)
Each step records agent, role, model, kind (`perception | model_turn | tool_call | deterministic | human`), prompt version (`pa-1`, `pr-1`, `pl-1`, `sam-1`), input/output summary, duration, status (`ok | error | tool_error`), API calls, retries, tokens, API latency and estimated cost. Totals add model turns, tool calls, tool errors and wall time. Tokens are `None` when the API returns no usage. Prices in `config.PRICE_TABLE_DEFAULT` are **placeholders** (`PRICES_VERIFIED = False`), so cost is an estimate of plumbing, not a bill.

## Failure handling
- HTTP: timeouts, 429 and 5xx are retried at most 2 times with exponential backoff (0.5 s, cap 8 s); `Retry-After` is honoured up to 30 s, otherwise fail fast. Other 4xx and non-JSON bodies are not retried. After that a `ProviderError` reaches the caller.
- Perception: calibration failure, no objects, no SAM mask, merged or low-fit masks, duplicate or missing identities block the run or need explicit acceptance. If the first photo has any issue, the agent is **deferred** until the person confirms (no model spend on unaccepted geometry).
- Recapture problems keep the run in `awaiting_recapture` with a retake message. A tool bug becomes an error tool result, not a crash.
- Input limits: image max 6000 px side, constraint text capped at 2000 chars and treated as data; the API key comes only from the environment.

## Evaluation harness (`evals/`, see `evals/README.md`)
`evals/run.py --provider mock|replay|meta --split dev|heldout` scores the whole pipeline against hand-written ground truth and writes `evals/reports/`.
- Write ground truth first, check it with `--validate-only`, tune on `dev` only. `heldout` is reported once per manifest and provider (marker file; `--allow-heldout-rerun` to override; mock exempt).
- `meta` requires `--max-calls` (counts HTTP attempts including retries) and stops with a partial report when it is spent.
- Ground truth never reaches a model: only photo, board size, check length, anchors and constraint text go in; an outgoing-request guard and `tests/test_evals.py` check this.
- Metrics: footprint edge error, constraint interpretation, route agreement (`false_open` must be 0), move-count optimality against an independent brute force, plan validity on the true layout, wrong-move detection, run cost.

## Verified vs UNVERIFIED
- **Verified here:** the offline suite passes (`pytest -q` in `apps/api`: 80 passed, 1 skipped, 2026-10-30). It covers calibration rejection, route logic, search against a brute force, locks, agent caps, `ask_user` pause, recovery, retry/fault injection, record/replay, telemetry, the eval leak guard, and replay of sanitized live recordings (`tests/fixtures/live/`).
- **Verified live 2026-10-30 (synthetic inputs, 11 billed calls across the smoke test):** base URL, `POST /v1/chat/completions` with `max_completion_tokens`, `reasoning_effort`, strict `json_schema` (needs `additionalProperties:false` and every property required), image input as `image_url` data URL, `tools` + `tool_choice:"auto"` (other values are HTTP 400), `tool_calls[].function.arguments` as a JSON string, `usage.prompt_tokens/completion_tokens`, published prices. SAM 3.1 is `POST /v1/responses` (`input_text` + `input_image`), reply is text records `<|box;...|><|mask;...data=H,W,PAYLOAD|>` in `output[0].content[0].text`; empty text = no match; no token usage, billed per image. See `providers/sam.py`.
- **Verified 2026-10-30, no new calls:** SAM mask decoding with Meta's `meta-sam-parser` 0.1.0 on both recorded live replies. One-bit and lossless give the same rasters, our record parser matches the library's stream parser, and a decoded mask becomes a metric footprint of the right size (`tests/test_live_shapes.py`). Real-photo mask quality is still NOT TESTED.
- **UNVERIFIED:** whether `temperature` is accepted (it is no longer sent), rate limits under load, and all accuracy on real photos.
- **NOT TESTED:** muse-spark-1.3 or sam-3.1 accuracy on real tabletop photos, whether Spark tool calling follows this protocol, real-photo footprint error, two-move search time on larger boards. Only a synthetic manifest exists; its held-out split shares the dev generator, so it proves plumbing, not generalisation. The real-model gate is `docs/planning/04-FEASIBILITY-TEST.md`.
- **Scope limits:** tabletop model only, not accessibility compliance. Rectangles only, orientation unchanged by moves, wheelchair turning radius not modelled, mock labels come from a colour table.
