# Core workflow

1. **Set up a measurable scene.** Photograph one planar tabletop layout from overhead. Supply a known board dimension or four measured calibration corners. Reject an uncalibrated perspective view for metric claims.
2. **Review object footprints.** Propose masks/rectangles for six objects. The user confirms dimensions, movable versus locked status and destinations. Mark uncertain footprints for correction.
3. **Specify a route and constraints.** Select entrance and destination; enter a clearance width as an explicit test parameter. Accept simple constraints such as 'keep this desk fixed' or 'move no more than two objects'. Show the parsed constraints for confirmation.
4. **Solve.** Compute whether the current route is connected under the chosen planar assumptions. Enumerate allowed rearrangements, with costs ordered by number of moved objects and then displacement.
5. **Show a concrete proposal.** Highlight moved objects, before/after locations and a route witness. State the search limits. If no solution exists within those limits, explain which constraint blocks it.
6. **Rearrange and capture again.** User moves the physical pieces and takes a second calibrated photo.
7. **Verify observed outcome.** Compare actual with expected footprints and recompute the route from the new observation. A clicked 'done' button is not evidence that the layout changed.
8. **Export.** Save before/after measurements, constraints, proposed changes, observed deviations and the route result. Ask for correction or remeasurement when uncertainty matters.

## First UI

Image with editable rectangles, a measured 2D plan, a small constraints panel and a move list. Keyboard controls edit object values and lock state. Use plain status: blocked, candidate plan, needs remeasurement, route verified under model assumptions. Avoid a universal accessibility score.

## Scope and impact

One flat board, six rectangles, no rotations initially, a fixed candidate placement grid and at most two moved objects. User-confirmed footprints make the experiment measurable. The potential beneficiary is an organizer trying to adjust a venue layout without moving everything. Actual venue/accessibility benefit remains untested.

Physical rooms introduce doors, slopes, thresholds, turning manoeuvres, furniture undercuts and human needs beyond this planar model. Do not label the tabletop result a venue certification. A meaningful follow-up would be organizer and accessibility-informed review after the hackathon prototype.

## Demo

Show a blocked route and a locked desk. The app proposes two or fewer moves, the user performs them, and the app verifies the second image. Then deliberately move one piece incorrectly: the app must flag the mismatch and recheck connectivity rather than replay the original green route.
