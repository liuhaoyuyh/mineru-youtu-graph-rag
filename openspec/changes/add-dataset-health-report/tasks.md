## Implementation Checklist
- [ ] Instrument `models/constructor/kt_gen.py` (and related retriever caches) to compute document, chunk, node, edge, and community counts plus token/runtimes at the end of each build.
- [ ] Serialize the metrics to `output/graphs/<dataset>_health.json` and add a config toggle for thresholds/warnings.
- [ ] Add backend routes (REST + WebSocket payload) that expose the health JSON and ensure responses degrade gracefully when metrics are missing.
- [ ] Update the frontend dashboard to render the health summary with warning badges/tooltips and provide a download control.
- [ ] Document how to interpret the health report in README/FULLGUIDE and add a smoke test script that hits the new endpoint after graph construction.
