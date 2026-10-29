# Test results — AccessShift

Recorded 29 October 2026. **Preparation complete; full product feasibility gate NOT TESTED.**

| Check | Actual result | What it establishes |
|---|---|---|
| Local core probe | **8/8 passed**; exit code 0 | Blocked/open connectivity, a minimum one-move solution within a declared candidate set, lock preservation, invalid placements and wrong-after-state detection. |
| Local client/schema probe | **6/6 passed**; exit code 0 | Prepared synthetic output accepted; unexpected/missing fields, invalid types/enums and bad semantic references or geometry rejected; default execution reads no key and makes no network request |
| Configuration dry run | Passed | Prompt/schema files load; no input image supplied; zero model calls |
| Real Meta inference | **NOT RUN** | User instructed “Prepare the tests first”; no live call was made |
| Complete product gate | **NOT TESTED** | Requires the real inputs and protocol in [04-FEASIBILITY-TEST.md](04-FEASIBILITY-TEST.md) |

## Scope of the evidence

Ideal discrete cells are supplied. No photographed footprint measurement, SAM segmentation, physical recapture or real accessibility usability has been tested. The probe searches zero/one move only; the proposed app may extend to two.

The hand-authored `fixture.synthetic.json` is validator input, not a captured model response. The local checks are small engineering probes, not benchmark results or evidence of adoption, time saved, field safety or winning odds.

Reproduce with `python spike/check_core.py` and `python spike/check_probe.py` from this project folder. Saved machine-readable output: [results/local-checks.json](../spike/results/local-checks.json). These scripts and reports may be published only under the eventual project/data rules; they do not constitute a submitted product.

## Next evidence needed

Prepare a measured tabletop scene and independent ground-truth layouts. Test image calibration and real footprints before adding the full planner interface.

Record every live attempt, failure and correction. Store model outputs under the local gitignored `spike/private/` directory until reviewed for sharing. Update this record after the real test; do not relabel a synthetic pass as a model pass.
