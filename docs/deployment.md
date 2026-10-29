# Deployment

## Docker (recommended)

```bash
docker build -t access-shift .
docker run --rm -p 8000:8000 access-shift
```

Open http://127.0.0.1:8000. With no `MODEL_API_KEY` the app runs in **mock** mode (offline pipeline, canned model replies; never reported as model results).

Compose: `docker compose up --build`. To use Meta models, copy `.env.example` to `.env`, set `MODEL_PROVIDER=meta` and `MODEL_API_KEY`, then start with `docker compose --env-file .env up`.

## Environment

| Variable | Meaning |
|---|---|
| `MODEL_PROVIDER` | `mock` (default without a key), `meta` (live, billed) or `replay` (offline) |
| `MODEL_API_KEY` | Meta Model API key. Never commit it. Server-side only. Not needed for `mock` or `replay`. |
| `RECORDINGS_DIR` | Where recordings live. Default `apps/api/demo_data/recordings/` (gitignored). |
| `RECORD_MODEL_CALLS` | `1` (default) records in `meta` mode; `0` disables. |

## Record and replay

- `MODEL_PROVIDER=meta`: every **successful** request/response pair is saved as `<sha256>.json` in `RECORDINGS_DIR`. The key is the sha256 of the normalized request (method, path, JSON body with long strings such as base64 images replaced by their digest). The `Authorization` header is never read into the key or written to disk (tested). Error responses (429/5xx) are not recorded.
- `MODEL_PROVIDER=replay`: the same adapters, but a transport answers from the recordings. No key, no network. A request with no recording raises `ReplayMissError` (not a `ProviderError`, so the agent's graceful-degradation paths do not swallow it) naming the key and folder. It never guesses.
- Use: do one billed `meta` run (cap it with `evals/run.py --max-calls`), then rerun the identical inputs under `replay` for demos, CI and prompt-independent regression checks. Changing a prompt, image or tool result changes the key and requires a re-record.
- Recordings are model output about your photos: keep them out of git. `git add -f` a single reviewed file only on purpose.
- The whole agent run is deterministic given the same inputs (server tools are deterministic, no clocks or ids go into prompts), which is what makes replay of the multi-turn tool loop possible.

## Limits

- Run **one process** (no `--workers`): the worker is in-process and state is local.
- Data lives in `/app/apps/api/demo_data`; the compose file mounts a volume there. Add retention and auth before a public host.
- Live Meta endpoint shapes are **UNVERIFIED** until the real-model gate runs. The image has been built and started only where stated in the README status.
