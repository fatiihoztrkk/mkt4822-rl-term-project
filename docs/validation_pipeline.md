# Validation Pipeline

Validation is the gate between staged JSON and real Python files.

## Staged checks

For every `CodeUnitDraftResponse`:

1. Reject forbidden code patterns.
2. Write code to a temporary workspace.
3. Run syntax validation.
4. Run isolated import smoke validation.
5. Run Ruff when installed/configured.
6. Run Pyright when installed/configured.
7. Run pytest when configured.
8. Run deterministic semantic checklist.
9. Store the draft and validation result as JSON.

If any primary failure remains, repair happens on the JSON draft. The real `algorithm.py` is not touched.

## Final checks

After a staged draft passes, the runtime creates a full-file patch for `algorithm.py`, applies path/security checks, writes the file, and runs the validation pipeline again. If final validation fails, the previous file content is restored.

## Failure taxonomy

- `SYNTAX`
- `IMPORT`
- `TYPE`
- `LINT`
- `TEST`
- `SEMANTIC`
- `PATCH`
- `SECURITY`
- `UNKNOWN`
