## ADDED Requirements
### Requirement: Graph Construction Health Report
Graph construction MUST emit a structured health report that captures ingestion metrics and warning flags for each dataset build.

#### Scenario: Metrics persisted after build
- **GIVEN** a dataset build completes (success or handled failure)
- **WHEN** the constructor finishes writing graph/chunk artifacts
- **THEN** a `<dataset>_health.json` file is saved with counts for processed documents, chunks, nodes, edges, communities, runtime, and token usage plus any warning flags.

#### Scenario: Missing metrics handled gracefully
- **GIVEN** a user requests health data for a dataset that has not been rebuilt since the feature shipped
- **WHEN** the backend cannot find the health JSON
- **THEN** it returns an empty report payload with `status="pending"` so the frontend can prompt the user to rebuild.

### Requirement: Surface Dataset Health to Users
The system MUST expose health report details through programmatic APIs and the web dashboard so operators can validate readiness before querying the graph.

#### Scenario: REST access to health metrics
- **GIVEN** a dataset name
- **WHEN** a client calls `GET /api/datasets/{name}/health`
- **THEN** the backend responds with the latest health report, HTTP 200, and warning codes when thresholds are breached.

#### Scenario: UI highlights warnings
- **GIVEN** a health report that contains warning flags (e.g., low coverage, high token cost)
- **WHEN** the frontend renders the dataset card
- **THEN** it displays a warning badge, textual summary, and a link/button to download the full report before enabling conversational queries.
