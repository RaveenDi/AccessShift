# First feasibility experiment — three-hour ceiling

## Hardest assumption

Two observations of a calibrated layout can produce sufficiently accurate footprints to propose and verify a real rearrangement, with less work than entering the layout manually. A path solver alone does not establish this.

## Inputs

One measured rectangular board, six movable rectangular pieces with simple labels, a ruler and overhead photos. These materials have not been confirmed available. One object must be locked. Prepare three known configurations and one held-out arrangement; include both an open and blocked route. Independently record dimensions and expected connectivity under the chosen clearance envelope.

Use only owned images. Prepare ground truth before running perception; never send the expected move or measured answer to the extraction model. Provide calibration dimensions as legitimate input. Limit claims to the table model.

## Protocol

| Elapsed | Work |
|---|---|
| 0–30 min | Calibrate image coordinates against board dimensions; check one unused known length |
| 30–80 min | Real Spark constraint extraction and SAM footprints; measure error before and after manual correction |
| 80–130 min | Solve blocked/unblocked cases with bounded search and fixed lock state; compare manual planning effort |
| 130–180 min | Test held-out layout, perform a move, recapture, then deliberately perform an incorrect move and recheck |

## Proposed pass gate (not achieved)

- All six pieces accounted for; uncertain/merged masks trigger correction. Choose tolerance before testing: for example, maximum edge error 5 mm on a 600 mm board. This is an engineering spike target, not a human-accessibility standard.
- Exact connectivity agreement with independent reference geometry for all four layouts under the explicit planar envelope. Borderline clearance is flagged rather than declared clear.
- The returned plan honors locks, bounds and non-overlap; its claimed move count matches exhaustive search of the declared small candidate set.
- One correct physical action is recognized, one deliberate wrong action is flagged, and the route is recomputed from recaptured observations.
- Compare setup/correction/planning time with manual rectangle entry; record all attempts. No meaningful model benefit means reconsider the concept even if geometry works.

## Prepared tools

`python access-shift/spike/check_core.py` checks finite-grid connectivity, move count, lock preservation, collision rejection and wrong-after-state detection. The fixture uses ideal supplied cells and is explicitly **not a metric camera test**.

The prepared image/text model probe can check label/box and constraint extraction later. It does not call SAM, calibrate masks or verify physical measurements. Those require the scene, segmentation adapter and protocol above. User requested tests be prepared first; no live call is performed now. Record actual results in [05-TEST-RESULTS.md](05-TEST-RESULTS.md).
