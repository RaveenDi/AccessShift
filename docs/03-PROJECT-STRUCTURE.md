# Proposed implementation structure

After this finalist is selected for implementation, use its own folder (`access-shift/`), which is its own separate Git repository. This folder contains planning and probe code only.

```text
access-shift/
  README.md
  AGENTS.md
  .env.example
  .gitignore
  pyproject.toml
  uv.lock
  apps/web/
    package.json
    package-lock.json
    src/
      features/capture/
      features/calibration/
      features/layout-review/
      features/constraints/
      features/move-plan/
      features/verification/
      components/
      api/                     # generated API types
  apps/api/app/
    main.py
    config.py
    api/
    domain/
      calibration.py
      footprints.py
      constraints.py
      connectivity.py
      search.py
      verification.py
    providers/
      meta_vision.py
      meta_segmentation.py
    prompts/
      layout_extract.v1.md
      constraints.v1.md
      schemas/
    workflows/
    storage/
    telemetry/
  apps/api/tests/
  evals/
    manifest.jsonl
    fixtures/                  # owned publishable photos and measured layouts
    ground_truth/              # manually checked; hidden from perception model
    run_layout_eval.py
    reports/
  docs/
    architecture.md
    measurement-assumptions.md
    limitations.md
  deployment/Dockerfile
  submission/
```

Separate measured geometry from model extraction. The domain modules must run on fixtures without any model call. Keep prompts/schemas versioned, model API calls behind adapters and resulting masks/poses traceable to captures. Use one container/service and one SQLite-backed worker for the demonstration, with persisted files and an explicit interrupted-job policy.

Build order: geometry probe → calibrated physical observation → model constraint/footprint extraction → bounded moves → recapture verification → minimal UI → unseen-layout evaluation → demo. The observed before/after loop matters more than a polished room editor.

Allow at most three hours for this candidate's first gate. If all finalists are probed, the shared allocation is 8 hours selection, 21 implementation, 6 integration/evaluation, 8 deployment/video/docs and 2 buffer = **45 hours**. See [shared budget](../../drawn-to-scale/docs/03-PROJECT-STRUCTURE.md). Do not independently spend that amount on all three candidates.
