# Rollback Procedure

Rollback is based on immutable image identifiers and mounted data volumes.

## Before Release

Record:

- CPICANN app image tag and digest;
- XDecomposer worker image tag and digest;
- Git commit SHA;
- model manifest hashes;
- XDecomposer manifest hashes;
- database/object-store or volume snapshot identifier, if used.

## Rollback Steps

1. Stop traffic at the reverse proxy or switch to maintenance mode.
2. Stop current containers.
3. Restore the previous compose file and environment.
4. Restore `models/` and `runs/` from the last known-good snapshot if needed.
5. Start the previous immutable image tags or digests.
6. Run:

```bash
docker compose config
docker compose ps
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
```

7. Run one known CPICANN single-phase sample.
8. If XDecomposer was enabled, run an approved multiphase smoke test.
9. Re-enable traffic only after checks pass.

## Non-Rollback Conditions

Do not mark rollback complete when:

- the previous model manifest hash differs from the expected value;
- run volume restoration failed;
- health checks pass but known sample output changed unexpectedly;
- HTTPS or authentication configuration was lost.
