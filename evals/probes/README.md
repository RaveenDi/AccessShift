# Prepared extraction probe

This is a small **image/text → structured JSON** probe, not the complete AccessShift application. It uses the documented [Meta image input](https://dev.meta.ai/docs/image-understanding) and [structured output](https://dev.meta.ai/docs/structured-output) request shapes. Live endpoint compatibility and task performance remain UNVERIFIED until a real call passes.

## Files

- `check_core.py`: local synthetic algorithm checks.
- `model_probe.py`: one request at most, **dry run by default**.
- `check_probe.py`: validates the local client and verifies its default path does not read a key or create a network client.
- `schema.json` and `prompt.md`: versioned extraction contract.
- `fixture.synthetic.json`: hand-authored example used only to test validation.
- `results/local-checks.json`: actual local results.
- `private/`: future owned inputs and model outputs; gitignored.

Python 3.12 is available in this workspace. Drawn to Scale's core probe additionally uses installed NumPy. The transport uses Python's standard library; loading a local `.env` uses `python-dotenv`, which is installed here. The local schema validator intentionally supports only the subset used by these three files; it is not a general JSON Schema implementation.

## Run local checks

From the planning-repo root:

```powershell
python access-shift/spike/check_core.py
python access-shift/spike/check_probe.py
python access-shift/spike/model_probe.py
python access-shift/spike/model_probe.py --validate-file access-shift/spike/fixture.synthetic.json
```

All commands above are local-only. The last command validates a synthetic object; it cannot score model quality.

## Later: real owned-input extraction

1. Put an owned PNG/JPEG and a short UTF-8 context/transcript file under this folder's `private/`. Files are not included yet. Keep evaluator answers elsewhere and never pass them as context.
2. Configure `MODEL_API_KEY` in the process environment or in `access-shift/.env`, using the blank `.env.example` as a template. Never paste the key into chat. Environment values take precedence over the local file.
3. Prepare the request without sending it:
```powershell
python access-shift/spike/model_probe.py --image access-shift/spike/private/input01.png --text-file access-shift/spike/private/context01.txt
```
4. When ready to consume API usage, repeat with `--live`:
```powershell
python access-shift/spike/model_probe.py --image access-shift/spike/private/input01.png --text-file access-shift/spike/private/context01.txt --live
```

The live command sends one request to the fixed Meta endpoint, refuses redirects, waits up to 45 seconds for the network operation and performs no automatic retry. Network timeout can leave the remote outcome unknown; it is not a cancellation guarantee. The probe caps input size and response bytes, but has not yet verified a provider token-budget parameter; review account usage before repeated live runs. It defaults to `muse-spark-1.3`, with `META_MODEL` available as a local override.

Output is written to a new file under `evals/probes/private/`; an explicit `--output` must stay there and cannot overwrite an existing file. The report contains response data, validity, timing, usage when returned, input hashes and prompt/schema hashes. It never saves the API key or request headers. Format validation is not a factual-accuracy check.

This probe returns proposed image boxes and constraints; it does **not** call SAM, produce calibrated masks or verify metric clearance. Those remain separate steps in the physical feasibility protocol.
