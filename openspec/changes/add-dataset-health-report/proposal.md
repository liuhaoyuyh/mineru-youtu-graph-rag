# Proposal: add-dataset-health-report

## Why
- Graph construction is currently a black box; users do not know how many documents, nodes, and relations were materialized or whether caches were reused.
- Troubleshooting failed builds requires digging through logs under `output/logs/`, slowing iteration when aligning schemas or tuning LLM prompts.
- Ops teams need a fast signal that the constructed graph is healthy enough to answer questions before exposing it to analysts.

## What Changes
- Instrument the constructor pipeline to emit per-dataset metrics (input files processed, chunks generated, node/edge counts, community coverage, average token use, runtime) plus warning flags when thresholds are missed.
- Persist the latest metrics in a lightweight JSON artifact under `output/graphs/<dataset>_health.json` and expose them via a new REST endpoint (`GET /api/datasets/{name}/health`) as well as the existing WebSocket progress channel.
- Extend the frontend dashboard with a "Dataset Health" card that summarizes the metrics, highlights warnings, and provides a "download full report" action.
- Document the workflow in the README/full guide so teams know how to interpret the health scores before releasing a dataset.

## Impact
- Introduces a new `graph-observability` capability spec; no existing specs are modified.
- Requires minor changes to the constructor, backend API, and frontend UI but does not alter the on-disk graph format.
- Adds a new JSON artifact per dataset and a read-only API surface; no migration is needed for existing data beyond generating the health report on the next build.
