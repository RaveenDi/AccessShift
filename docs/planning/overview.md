# AccessShift

Status: **user-selected finalist; architecture proposed; real-model spike pending**. Updated 2026-10-29. Planning and small feasibility checks only; no app deployment or separate Git repository created.

**Promise:** given a measured layout and confirmed constraints, suggest a small rearrangement that opens a specified route, then verify the observed changes. Initial proof uses a calibrated tabletop model. It does not establish real-world accessibility compliance or usability.

| Read | Purpose |
|---|---|
| [01-WORKFLOW.md](01-WORKFLOW.md) | Capture, confirm, propose moves, reobserve |
| [02-ARCHITECTURE.md](02-ARCHITECTURE.md) | Model perception plus deterministic planning/verification |
| [03-PROJECT-STRUCTURE.md](03-PROJECT-STRUCTURE.md) | Proposed app layout and implementation sequence |
| [04-FEASIBILITY-TEST.md](04-FEASIBILITY-TEST.md) | Three-hour physical/model gate |
| [05-TEST-RESULTS.md](05-TEST-RESULTS.md) | Executed local checks and pending physical evidence |

Run local checks: `python access-shift/spike/check_core.py`. This tests a discrete layout planner on synthetic geometry. It does not test camera measurements, segmentation or wheelchair usability.

The budget is shared: **up to 45 hours for selection plus one delivered project**, not three separate 45-hour builds. See the shared allocation. This product is a separate alternative to the mapping finalists.

Sources: idea card, prior research, scorecard. Existing [RoomSketcher accessibility planning features](https://help.roomsketcher.com/hc/en-us/articles/36420331951773-Can-I-Create-Accessible-Floor-Plans-in-the-RoomSketcher-App) mean our contribution must be demonstrated through measured intervention and rechecking, not a claim to have invented accessible layout planning.
