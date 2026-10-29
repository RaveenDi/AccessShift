# Agent architecture (as built tonight)

Scope: a calibrated **tabletop model**. Not accessibility compliance, not a real-world measurement.
Principle: **models perceive and interpret; deterministic code decides connectivity.**

## Hierarchy (one pipeline, human in the loop)
`Perceiver -> Calibrator -> Constraint Interpreter -> Route Checker -> Human confirm -> Planner -> (person moves objects) -> Perceiver (recapture) -> Calibrator -> Verifier`

| Agent | Kind | Meta product (live mode) | Mock mode (default, no key) |
|---|---|---|---|
| Perceiver | model | `sam-3.1` masks + `muse-spark-1.3` labels, board corners, check reference | OpenCV colour segmentation on the synthetic image |
| Calibrator | deterministic | none | same code |
| Constraint Interpreter | model | `muse-spark-1.3` structured output into `ConstraintSpec` | regex rule parser |
| Route Checker / Planner | deterministic | none | same code |
| Human confirm | person | none | API call |
| Verifier | deterministic | none (consumes a fresh Perceiver result) | same code |

Every step is logged `{agent, role, model, input_summary, output_summary, duration_ms, status}`; `GET /api/runs/{id}/trace`.
Run states: measuring -> needs_confirmation -> planning -> awaiting_recapture -> verifying -> ready | needs_remeasurement | failed.

## Deterministic vs model boundary
- Model output is data. It is schema-validated (`ConstraintSpec`), range-checked (clearance 20-300 cm, locked labels must be known labels) and never reaches the geometry unvalidated.
- Pixels become centimetres only through the homography from 4 board corners; an independent known length must agree within 3% or the run is rejected.
- Connectivity: obstacles expanded by clearance/2 + uncertainty margin, boundary shrunk, 4-connected centre path on a 5 cm raster. Result is `open` (open at radius+margin), `blocked` (blocked even at radius-margin) or `borderline` (never accepted as clear). A witness path is cross-checked against continuous polygon distance.
- Planner: finite grid (default 10 cm) of same-orientation placements, locks respected, 0 then 1 then 2 moves, minimum total displacement. "Minimal" is claimed only for the enumerated set and only if enumeration finished; otherwise `search_incomplete` / `plan_found_search_incomplete`; exhaustion gives `no_solution_in_set`.
- Verifier: recapture is perceived and calibrated independently (the plan never supplies positions). Per object: `moved_as_planned | unchanged | not_moved | moved_to_wrong_place | unexpected_move`, then the route is recomputed from observed geometry. Verdicts: `verified_open`, `route_open_but_deviates_from_plan`, `route_still_blocked`, `route_borderline`, `needs_remeasurement`.

## Guardrails
- Mock providers are the default without `MODEL_API_KEY`; no live call exists in tests. Key only from env/`.env`, never logged.
- Ill-conditioned calibration, ambiguous/duplicate/missing identities, low-fit masks (need explicit acceptance) block the run.
- Human must confirm footprints and locks before planning. Endpoints without a user-marked anchor are refused.
- Model-only claims of "accessible" are impossible by construction: no code path lets a provider set route status.
- Upload size and image size limits; constraint text capped at 2000 chars and treated as data.

## Evaluation plan (NOT yet run)
1. Real-photo gate in `docs/04-FEASIBILITY-TEST.md`: footprint error vs ruler (target 5 mm on 600 mm board), connectivity agreement on 4 layouts, correct and wrong physical move.
2. Constraint extraction vs hand-written expected schema (Spark).
3. Manual-rectangle baseline: time to correct model footprints vs drawing them.
4. Never send expected moves or ground truth to a model.

## Verified vs UNVERIFIED
- Verified here (synthetic image, mock providers): calibration rejection, route logic, search optimality on the declared set vs an independent brute force, lock preservation, correct vs wrong move detection, trace, API end to end (`pytest -q`).
- UNVERIFIED: every Meta call. Endpoint `https://api.meta.ai/v1` + OpenAI-style chat shape come from planning notes; the `response_format` parameter, the image-in-chat message shape, the SAM 3.1 route `/images/segmentations` and its mask encoding are guesses behind `app/providers/meta.py`. SAM/Spark accuracy on real tabletop photos: **NOT TESTED**. Two-move search time on larger boards: only budgeted, not benchmarked.
- Known simplifications: rectangles only; orientation unchanged by moves; wheelchair turning radius not modelled; mock labels come from a colour table.
