"""Local trust-boundary checks for the prepared client; networking is blocked."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import model_probe as probe

HERE = Path(__file__).resolve().parent


def accepts(data, schema):
    try:
        probe.validate(data, schema)
        probe.semantic_checks(data, schema["title"])
        return True
    except (ValueError, TypeError):
        return False


def main():
    schema = json.loads((HERE / "schema.json").read_text())
    fixture = json.loads((HERE / "fixture.synthetic.json").read_text())
    extra = {**fixture, "invented_field": True}
    missing = deepcopy(fixture)
    del missing[next(iter(schema["properties"]))]
    semantic = deepcopy(fixture)
    invalid_type = deepcopy(fixture)
    name = schema["title"]
    if name == "sketch_graph":
        semantic["relations"] = [{"from_id": "p1", "to_id": "not-a-landmark", "kind": "near", "evidence": "test"}]
        invalid_type["landmarks"][0]["x"] = True
    elif name == "combined_observations":
        semantic["conflicts"][0]["observation_ids"] = ["o1", "missing"]
        invalid_type["observations"][0]["source"] = "invented_modality"
    else:
        semantic["objects"][0]["bbox"] = [0.8, 0.2, 0.3, 0.5]
        invalid_type["objects"][0]["bbox"][0] = True
    captured = io.StringIO()
    with patch.object(sys, "argv", ["model_probe.py"]), patch.object(probe, "load_key", side_effect=AssertionError("Unexpected credential read")), patch.object(probe.urllib.request, "build_opener", side_effect=AssertionError("Unexpected network")), contextlib.redirect_stdout(captured):
        code = probe.main()
    dry_result = json.loads(captured.getvalue())
    checks = {
        "hand_authored_fixture_valid": accepts(fixture, schema),
        "unexpected_field_rejected": not accepts(extra, schema),
        "missing_required_field_rejected": not accepts(missing, schema),
        "bad_semantic_reference_or_geometry_rejected": not accepts(semantic, schema),
        "bad_type_or_enum_rejected": not accepts(invalid_type, schema),
        "default_reads_no_key_and_sends_no_request": code == 0 and dry_result["model_calls"] == 0 and dry_result["mode"] == "dry_run",
    }
    print(json.dumps({"kind": "local_client_validation_only", "checks": checks,
                      "passed": sum(checks.values()), "total": len(checks), "model_calls": 0}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
