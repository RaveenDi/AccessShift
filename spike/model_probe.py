"""One image/text extraction request, dry-run by default. No tools or agent loop.

Docs: https://dev.meta.ai/docs/image-understanding
      https://dev.meta.ai/docs/structured-output
This same small transport is copied into each finalist to keep probes independent.
The local validator supports only the schema subset authored for these probes.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
API_URL = "https://api.meta.ai/v1/chat/completions"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def validate(value, schema, path="$ "):
    """Validate our explicit types/objects/arrays/enums/ranges; not general JSON Schema."""
    typ = schema["type"]
    checks = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": type(value) is bool,
        "number": type(value) in (int, float) and math.isfinite(value),
        "integer": type(value) is int, "null": value is None,
    }
    if not checks.get(typ, False):
        raise ValueError(f"{path}: expected {typ}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unsupported enum value")
    if typ == "object":
        props = schema["properties"]
        if set(value) - set(props) or set(schema["required"]) - set(value):
            raise ValueError(f"{path}: unexpected or missing keys")
        for key, item in value.items():
            validate(item, props[key], path + "." + key)
    elif typ == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 1000):
            raise ValueError(f"{path}: array length outside bounds")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
    elif typ == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 10000):
            raise ValueError(f"{path}: string length outside bounds")
    elif typ in ("number", "integer"):
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path}: number outside bounds")


def semantic_checks(value, name):
    if name == "sketch_graph":
        ids = [item["id"] for item in value["landmarks"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate landmark IDs")
        if any(r["from_id"] not in ids or r["to_id"] not in ids for r in value["relations"]):
            raise ValueError("Relation refers to an unknown landmark")
        compass = {"north_of", "south_of", "east_of", "west_of"}
        if value["north_direction"] == "unknown" and any(r["kind"] in compass for r in value["relations"]):
            raise ValueError("Compass relationship without stated orientation")
    elif name == "combined_observations":
        ids = [item["id"] for item in value["observations"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate observation IDs")
        for conflict in value["conflicts"]:
            references = conflict["observation_ids"]
            if len(set(references)) < 2 or not set(references).issubset(ids):
                raise ValueError("Conflict lacks two known observation references")
    elif name == "layout_observations":
        ids = [item["id"] for item in value["objects"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate object IDs")
        for obj in value["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            if x1 >= x2 or y1 >= y2:
                raise ValueError("Invalid bounding-box corner order")
        if any(c["target_id"] and c["target_id"] not in ids for c in value["constraints"]):
            raise ValueError("Constraint refers to an unknown object")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_key():
    env_file = HERE.parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ValueError("Install python-dotenv to load .env, or set MODEL_API_KEY in the process environment") from exc
        load_dotenv(env_file, override=False)
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise ValueError("MODEL_API_KEY is not configured locally; no request sent")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--text", default="", help="Non-sensitive example context only; prefer --text-file")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--schema", type=Path, default=HERE / "schema.json")
    parser.add_argument("--prompt", type=Path, default=HERE / "prompt.md")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-file", type=Path, help="Validate a prepared JSON object locally")
    parser.add_argument("--live", action="store_true", help="Send exactly one real API request; consumes API usage")
    args = parser.parse_args()
    schema_bytes = args.schema.read_bytes()
    schema = json.loads(schema_bytes.decode("utf-8-sig"))
    name = schema["title"]
    prompt_bytes = args.prompt.read_bytes()
    if args.validate_file:
        if args.live:
            raise ValueError("--validate-file is local-only; remove --live")
        data = json.loads(args.validate_file.read_text(encoding="utf-8-sig"))
        validate(data, schema)
        semantic_checks(data, name)
        print(json.dumps({"mode": "local_validation", "valid": True, "schema": name, "model_calls": 0}))
        return 0
    task_text = args.text_file.read_text(encoding="utf-8-sig") if args.text_file else args.text
    if len(task_text) > 12000:
        raise ValueError("Probe text limit is 12000 characters")
    content = [{"type": "text", "text": task_text or "Extract only what is supported by the supplied input."}]
    meta = {"schema": name, "schema_sha256": digest(schema_bytes), "prompt_sha256": digest(prompt_bytes),
            "text_sha256": digest(task_text.encode()), "image_supplied": bool(args.image)}
    if args.image:
        raw = args.image.read_bytes()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Probe image limit is 8 MiB")
        mime = "image/png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg" if raw.startswith(b"\xff\xd8\xff") else None
        if mime is None:
            raise ValueError("Only PNG/JPEG input is supported")
        meta["image_sha256"] = digest(raw)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(raw).decode()}})
    if not args.live:
        print(json.dumps({**meta, "mode": "dry_run", "model_calls": 0,
                          "status": "prepared" if args.image or task_text else "configuration_only",
                          "note": "No credentials loaded and no network request sent"}, indent=2))
        return 0
    if not args.image:
        raise ValueError("These three extraction probes require an image for a live run")
    if name == "combined_observations" and not task_text.strip():
        raise ValueError("Combined probe requires transcript text")
    key = load_key()
    model = os.environ.get("META_MODEL", "muse-spark-1.3")
    output = args.output or HERE / "private" / (datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ") + ".json")
    private_root = (HERE / "private").resolve()
    if not output.resolve().is_relative_to(private_root):
        raise ValueError("Keep model output under this probe's gitignored spike/private directory")
    if output.exists():
        raise ValueError("Output already exists; choose a new filename")
    request_body = {"model": model,
                    "messages": [{"role": "system", "content": prompt_bytes.decode("utf-8-sig")},
                                 {"role": "user", "content": content}],
                    "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}
    request = urllib.request.Request(API_URL, data=json.dumps(request_body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=45) as response:
            raw_response = response.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Meta API returned HTTP {exc.code}; response body omitted. No automatic retry.") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("Network/timeout error; outcome may be unknown. No automatic retry.") from None
    if len(raw_response) > 2 * 1024 * 1024:
        raise ValueError("Response exceeded probe limit")
    response_data = json.loads(raw_response)
    elapsed = round(time.monotonic() - started, 3)
    valid, error, result = False, None, None
    try:
        result = json.loads(response_data["choices"][0]["message"]["content"])
        validate(result, schema)
        semantic_checks(result, name)
        valid = True
    except (KeyError, IndexError, TypeError, ValueError):
        error = "Response failed JSON, schema or semantic validation; inspect private response and input manually"
    report = {**meta, "mode": "live", "requested_model": model, "returned_model": response_data.get("model"),
              "response_id": response_data.get("id"), "elapsed_seconds": elapsed, "model_calls": 1,
              "usage": response_data.get("usage"), "valid": valid, "error": error,
              "result": result, "response": response_data}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"mode": "live", "valid": valid, "elapsed_seconds": elapsed,
                      "report": str(output), "model_calls": 1}))
    return 0 if valid else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        # Never include request headers, credentials or arbitrary server error bodies.
        print(json.dumps({"error": str(error), "success": False}))
        raise SystemExit(2)
