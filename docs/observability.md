# Observability (placeholder, not built yet)

Users only see the severity score. Everything below is for us, to watch the live model.

## Plan

`serve.py` exposes Prometheus metrics at `GET /metrics` (via `prometheus_client`), protected by a
bearer token from an env var. Grafana Cloud (free tier) scrapes it with its **Metrics Endpoint**
integration (HTTPS + bearer token required, scrapes every 60 s) and we build the dashboard there.

## Metrics to track

| Metric | Type | Why |
|---|---|---|
| requests by outcome (`ok`, `bad_image`, `too_large`, `no_model`) | counter | traffic and failure rate |
| scoring latency | histogram | is the free instance fast enough |
| reconstruction error | histogram | drift: compare with the healthy/diseased ranges from `calibration.json` |
| severity score | histogram | what users are getting |
| within healthy range | counter | share of uploads that look healthy |
| error concentration (share of error in the top 10% of pixels) | histogram | the heat map as a number: high = localised lesions, low = diffuse error (lighting, background, not a leaf) |
| model version (`data_version`) | info gauge | which model is live |

Only numbers are recorded. Photos and heat maps are never stored or sent anywhere, as the proposal promises.

## Open questions
- A Grafana scrape every 60 s keeps the free Render instance awake (fine within 750 h/month, but it no longer sleeps).
- Counters reset when Render restarts the instance; Prometheus handles that, but short-lived spikes can be missed.
