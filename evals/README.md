# Evaluation harness

Scores the whole pipeline (perceive, calibrate, agent plans, human confirm, recapture, verify, recover) against
hand-written ground truth and writes `evals/reports/<stamp>-<split>.{json,md}`.

```bash
# from the repo root, with the API virtualenv
apps/api/.venv/Scripts/python scripts/make_demo.py            # synthetic photos + evals/manifests/demo.synthetic.jsonl
apps/api/.venv/Scripts/python evals/run.py --provider mock --manifest evals/manifests/demo.synthetic.jsonl --split dev
apps/api/.venv/Scripts/python evals/run.py --provider mock --manifest evals/manifests/demo.synthetic.jsonl --split heldout
apps/api/.venv/Scripts/python evals/run.py --provider mock --manifest <m> --split dev --validate-only   # check ground truth vs geometry
# real photos, after the owner writes evals/manifest.jsonl (default path):
python evals/run.py --provider meta   --split dev --max-calls 60      # live, billed, records to evals/recordings/
python evals/run.py --provider replay --split dev                      # offline, from those recordings
```

`--provider mock` is **plumbing only** (colour segmentation + a scripted policy on synthetic rectangles). Its
numbers say nothing about Meta models; every report states this at the top. `meta` refuses to start without
`--max-calls` (counts HTTP attempts, retries included; the run stops and the partial report says so).

## Discipline (the point of this folder)
1. **Write ground truth BEFORE running.** Measure the real objects with a ruler, write the manifest line, run `--validate-only`
   (checks your `route_open` / `expected_min_moves` against the deterministic geometry), commit the manifest, then run.
2. **Tune on `dev` only.** `heldout` is run once per manifest and provider: a second run is refused
   (`reports/.heldout-<manifest>-<provider>` marker) unless `--allow-heldout-rerun`. Mock is exempt. Never change
   prompts, thresholds or tools after looking at held-out results and then re-report them.
3. **Ground truth never reaches a model.** The pipeline receives only photo, board size, check length, anchors and
   constraint text (`Case.model_inputs`). An outgoing-request guard rejects forbidden field names and per-case secret
   strings (case id, `ground_truth.notes`); `tests/test_evals.py` inspects every request the fake Meta server sees.
4. Replay answers only from recordings and fails loudly on a miss. Do not edit recordings by hand.

## Manifest line (JSONL, one case per line, `#` comments allowed)
```json
{"id": "kitchen-1", "split": "dev", "photo": "evals/photos/kitchen-1.jpg", "board_mm": [600, 450],
 "check_len_mm": 200, "anchors_mm": {"door": [50, 225], "table": [560, 230]},
 "constraint_text": "Keep a 90 cm route from the door to the table. Lock the sofa.",
 "traps": ["blocked_route"],
 "answers": ["100 cm"],
 "ground_truth": {
   "footprints_mm": [{"id": "sofa", "label": "sofa", "x": 160, "y": 20, "w": 180, "h": 90}],
   "constraint": {"clearance_mm": 900, "start": "door", "end": "table", "locked_ids": ["sofa"], "max_moves": 2},
   "route_open": false, "route_expected_status": "blocked", "expected_min_moves": 1,
   "recapture": [{"photo": "evals/photos/kitchen-1-after.jpg", "expected": "correct"}],
   "expect_review": false, "notes": "free text, never sent to a model"}}
```
`board_mm`, `check_len_mm`, `anchors_mm` are inputs the person supplies in the UI (not ground truth). `footprints_mm` may use
`polygon_mm` instead of a rectangle; ids should equal the pipeline ids (label, then `label_2`). `expected_min_moves: null` means
no solution exists within the candidate set (the agent should say so). `answers` are canned replies if the agent asks the
person a question. For `borderline_clearance` cases `route_open` is the physical truth and `route_expected_status` is
`borderline` (the system must not certify a route inside measurement uncertainty); `expected_min_moves` then counts moves to a
route that is open with margin. `expect_review: true` means the correct behaviour is to stop and flag the footprint for a person.

## Trap cases to include (the synthetic manifest has one of each)
blocked route; already-open route; locked object that would be the obvious mover (honest "no solution"); deliberately wrong
move (nudged, plus an unrelated object moved) and a move that opens the route elsewhere (deviates from the plan); an unchanged
recapture; borderline clearance (a gap a few cm above the requirement); merged pieces (touching same-colour objects);
ambiguous width or object identity (`answers`). For real photos add: glare, a shadow across the board marker, a partly
occluded object.

## Metrics (all in the report)
- **Footprint edge error (mm):** worst-edge (Hausdorff) distance between perceived and true outline per matched object; also
  within-5-mm share, missed and spurious objects. Excludes `expect_review` cases.
- **Constraint interpretation:** clearance within 1 cm, endpoints, lock set vs `ground_truth.constraint`.
- **Route agreement:** initial route status (open only counts as open) vs `route_open`. `false_open` (said open, physically
  not) is the unsafe error and should be 0; `conservative_not_open` is borderline/blocked for a physically open route.
- **Move-count optimality:** plan size vs `expected_min_moves`; vs an independent brute force (`bruteforce.py`: exact for 0
  and 1 moves on a 40 cm grid, otherwise "more than one"); and **plan valid on ground-truth layout** (the moves are applied to
  the true footprints and the route is re-checked, so perception error cannot hide a bad plan).
- **Wrong-move detection:** wrong recaptures flagged (verdict not `verified_open`), correct ones accepted, and whether the
  recovery turn produced a corrective plan when the route was left blocked or borderline.
- **Cost of the run:** agent tool calls, model turns, API calls, retries, tokens (None if not returned), API and wall latency,
  estimated cost from UNVERIFIED placeholder prices.

## Honest limits
Synthetic results prove plumbing. The held-out synthetic split uses the same generator as dev. A plan that is `open` by a thin
margin can come back `borderline` after recapture because of measurement noise (seen in the mock held-out run); the verifier
refuses to certify it, which is intended, but a planner that prefers more slack is an open improvement (not tuned here).
Brute force does not verify two-move optimality independently.
