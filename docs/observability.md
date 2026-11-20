# Watching the live model (planned, not built yet)

Farmers only ever see the disease and its severity. This page is about what I want to watch behind the scenes once
the app is live, so I can tell whether the model is still behaving.

## The plan

`serve.py` will expose Prometheus metrics at `GET /metrics` using `prometheus_client`, protected by a bearer token
kept in an environment variable. Grafana Cloud's free tier can read that endpoint with its Metrics Endpoint
integration (it needs HTTPS and a bearer token, and checks every 60 seconds), and I'll build the dashboard there.

## What to track

| Metric | Type | Why it matters |
|---|---|---|
| requests by outcome (`ok`, `bad_image`, `too_large`, `no_model`) | counter | how much traffic there is and how often things fail |
| scoring latency | histogram | whether the free instance is fast enough |
| reconstruction error | histogram | drift: compare it with the healthy and diseased ranges in `calibration.json` |
| severity score | histogram | what users are actually being told |
| within healthy range | counter | the share of uploads that look healthy |
| error concentration (share of the error in the top 10% of pixels) | histogram | the heat map as one number: high means localised lesions; low means the error is spread out, which points to lighting, background or a photo that isn't a leaf |
| model version (`data_version`) | info gauge | which model is live |

Only numbers are recorded. As the proposal promised, photos and heat maps are never stored or sent anywhere.
