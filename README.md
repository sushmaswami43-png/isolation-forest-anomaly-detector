# Interactive Isolation Forest Anomaly Detector

A small web app for finding outliers in tabular data. Upload a CSV (or pick an example),
move two sliders, and the scatter plot, score histogram and anomaly table update without a
page refresh.

- **Backend:** Flask + scikit-learn `IsolationForest` (stateless JSON API)
- **Frontend:** plain JavaScript + Plotly.js (loaded from a CDN, so the browser needs internet access)
- **Deploy:** Procfile, Dockerfile and `render.yaml` included

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # http://localhost:5000
```

The app opens with an example already loaded. Try it:

1. Drag **Contamination** and watch the red points and the dashed threshold line move.
2. Drag **Number of trees** and note how the scores stabilise.
3. In *Features & plot*, untick a column to exclude it from the model, or change the X/Y axes.
4. Choose another example, or upload your own CSV.
5. Click **Download all results** to get your data plus `anomaly_score` and `is_anomaly`.

## Example datasets (`examples/`)

Synthetic, generated with a fixed seed by `examples/generate_examples.py` and with no labels,
so you can judge the result by eye. Each hides a small group of planted outliers.

| File | Columns | Planted outliers |
|---|---|---|
| `transactions.csv` (1,000 rows) | amount, seconds since last purchase | 40 large, rapid-fire purchases |
| `network_traffic.csv` (1,200 rows) | bytes, packets/s, duration | 60 short, heavy bursts |
| `sensor_quality.csv` (600 rows) | temperature, vibration, pressure | 30 hot, high-vibration, low-pressure parts |

## Using your own CSV

- A header row is required. At least two numeric columns and 10 complete rows are needed.
- Non-numeric columns are ignored; rows with missing or infinite values are dropped (the app reports how many).
- Limits: 5 MB and 50,000 rows.

## How the parameters change the result

Measured on `transactions.csv` (40 of 1,000 rows are planted outliers, i.e. a true rate of 4%):

| Contamination | Rows flagged | Precision | Recall |
|---|---|---|---|
| 0.01 | 10 | 0.70 | 0.17 |
| 0.04 | 40 | 0.78 | 0.78 |
| 0.10 | 100 | 0.40 | 1.00 |
| 0.20 | 200 | 0.20 | 1.00 |

- **Contamination** is the share of rows you expect to be anomalous. It does not change the
  scores; it only sets the cut-off, so about that fraction of rows is flagged. Too low misses
  outliers (low recall); too high flags normal rows (low precision). When unsure, use the score
  histogram: a natural gap in the right tail suggests where the threshold belongs.
- **Number of trees** (`n_estimators`) does not move the cut-off; it makes the scores more
  stable. On the same data, the average per-row score variation across random seeds fell from
  0.021 with 10 trees to 0.008 with 100 and 0.0035 with 500. Gains flatten while run time grows
  roughly linearly, so 100 to 200 is usually enough.

Scores are `-score_samples()`: higher means more anomalous. The app fixes `random_state=42`,
so the same data and settings always give the same answer. Isolation Forest does not need
feature scaling, so none is applied.

## API

| Method and path | Purpose |
|---|---|
| `GET /` | the web UI |
| `GET /healthz` | health check, returns `{"status": "ok"}` |
| `GET /api/examples` | list example datasets |
| `GET /api/examples/<id>` | download an example CSV |
| `POST /api/detect` | multipart form: `file` (CSV), `contamination` (0.001 to 0.5), `n_estimators` (10 to 500), optional `features` (comma-separated column names) |

`POST /api/detect` returns `scores`, `labels` (1 = anomaly), `threshold`, `n_rows`,
`n_anomalies`, the numeric `columns` and their `data`, plus `dropped_rows` and
`ignored_columns`. Errors come back as `{"error": "..."}` with status 400 (or 413 for files over 5 MB).

```bash
curl -F file=@examples/sensor_quality.csv -F contamination=0.05 -F n_estimators=100 \
     http://localhost:5000/api/detect
```

The server keeps no state: the browser re-sends the file with each slider change (debounced,
and stale requests are cancelled). That keeps the app simple to scale and deploy.

## Tests

```bash
python -m unittest discover -s tests -v
```

13 tests cover outlier recovery on planted data, contamination behaviour, reproducibility,
CSV cleaning, input validation, every API route and the upload size limit.

## Deployment

The app reads the `PORT` environment variable and exposes `app:app` for gunicorn. The
configuration files below are provided but I have not deployed to these platforms myself, so
check each platform's current UI and pricing.

**Render (simplest).** Push the project to GitHub, create a *New > Blueprint* (it reads
`render.yaml`) or a *Web Service* with build command `pip install -r requirements.txt` and start
command `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 60`. Set the health check
path to `/healthz`.

**Heroku or Railway.** Both detect the `Procfile` and `requirements.txt` automatically. For
Heroku: `heroku create && git push heroku main`.

**Docker / Google Cloud Run / any container host.**

```bash
docker build -t iforest-app .
docker run -p 8080:8080 iforest-app                 # http://localhost:8080
gcloud run deploy iforest-app --source . --allow-unauthenticated   # Cloud Run
```

Notes for production: `MAX_CONTENT_LENGTH` caps uploads at 5 MB; the free tiers of most hosts
have limited memory, so keep the row limit modest; add authentication if the app will hold
sensitive data (uploads are processed in memory and never stored).

## Project layout

```
app.py              Flask routes and error handling
detector.py         CSV loading, validation and Isolation Forest (no Flask imports)
templates/          index.html
static/             app.js (Plotly + controls), style.css
examples/           three CSVs + generate_examples.py
tests/              unit and API tests
Procfile, Dockerfile, render.yaml, requirements.txt
```
