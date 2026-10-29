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
| `MODEL_PROVIDER` | `mock` (default) or `meta` |
| `MODEL_API_KEY` | Meta Model API key. Never commit it. Server-side only. |

## Limits

- Run **one process** (no `--workers`): the worker is in-process and state is local.
- Data lives in `/app/apps/api/demo_data`; the compose file mounts a volume there. Add retention and auth before a public host.
- Live Meta endpoint shapes are **UNVERIFIED** until the real-model gate runs. The image has been built and started only where stated in the README status.
