# Patent Emerging Radar

Forecasts which technology areas will explode in the next 2–3 years by clustering 5.6M USPTO patent titles and tracking cluster growth dynamics with LSTM.

**Why patents?** Companies file patents 1–3 years before papers are published. A cluster of semantically similar patents accelerating today is a leading signal of the next CRISPR or LLMs — before academic literature catches up.

---

## Results

| Metric | 2-year | 3-year |
|---|---|---|
| Spearman ρ (test) | **0.42** | **0.47** |
| Spearman ρ (val) | 0.41 | 0.46 |
| Baseline (momentum) | −0.14 | −0.12 |
| LSTM vs. baseline | **+0.56** | **+0.59** |

No overfitting: test ≈ val. Naive momentum is *negatively* correlated — the LSTM genuinely learns structure the trend line misses.

---

## Pipeline

```
USPTO patent titles (g_patent.tsv, 5.6M patents 2002–2025)
    ↓
Embed — all-MiniLM-L6-v2 → 384-dim vectors, one .npz per year
    ↓
Cluster — UMAP (384d→10d) + HDBSCAN on rolling 2-year windows
          Cross-window stitching by centroid cosine similarity (≥0.85)
    → 2,006 stable technology clusters, auto-labelled by TF-IDF/NMF
    ↓
Features — 18 time-series signals per cluster per quarter
           (log_count, velocity, acceleration, jerk, momentum 4Q/8Q,
            rolling mean/std, above_trend, MACD, z-score, rank_pctl,
            log_cumsum, seasonality sin/cos)
    ↓
Forecast — GRU vs LSTM, 8Q lookback → 8Q/12Q growth delta
           Loss: Huber(δ=1.0). Metric: Spearman rank correlation.
           Train 2005–2017 / Val 2018–2020 / Test 2021–2023 (locked)
    ↓
Leaderboard — ranked bar chart of clusters predicted to erupt next 2 years
```

**LSTM won both horizons** and is the production model.

---

## Validation

Ground truth: 62 objective eruption events from 4,516 OpenAlex topics — no hand-picking.

- **Type A (emerging):** <500 papers/yr → ≥5× growth in 5 years
- **Type B (acceleration):** <10k papers/yr → ≥4× growth in 5 years
- Nobel cross-check: T11289 (single-cell genomics, Nobel 2024) confirmed at 4.5× growth 2014→2019

Retrospective lead time: patent cluster takeoff precedes publication eruption by a median of several quarters.

> Do **not** use `data/processed/validation/panel.csv` for validation — it contains 37 hand-picked topics with selection bias. Use `eruption_ground_truth.csv`.

---

## Dashboard

```bash
cd dashboard
python app.py
# → http://localhost:8050
```

Features a 3D technology galaxy (every cluster a star, sized by patent volume, coloured by forecast growth), click-through to cluster keywords + trend chart + real patent titles, domain filters, and tabs for metrics and pipeline explainers.

---

## Setup

```bash
pip install -r requirements.txt
```

### Running the pipeline

**Step 1 — Embed** (GPU recommended; or run locally)

```bash
# Local CPU (~8–12 min/year)
python scripts/embed_patents.py --years 2002 2025 --batch-size 256
# Output: data/processed/embeddings/YYYY.npz
```

For faster embedding, use `notebooks/kaggle_embed.ipynb` on a Kaggle T4 GPU (~4 min/year).

**Step 2 — Cluster + train + leaderboard**

Open `notebooks/patent_emerging_radar.ipynb` and run Parts 0–10.  
Or run the full Kaggle GPU pipeline: `notebooks/kaggle_pipeline.ipynb`.

---

## Key files

| Path | Purpose |
|---|---|
| `notebooks/patent_emerging_radar.ipynb` | Main notebook — full pipeline (Parts 0–10) |
| `notebooks/kaggle_embed.ipynb` | Kaggle T4 GPU embedding job |
| `notebooks/kaggle_pipeline.ipynb` | Full GPU pipeline (cluster + train + leaderboards) |
| `notebooks/kaggle_train.ipynb` | Training-only re-run from pre-saved cluster files |
| `scripts/embed_patents.py` | Local CPU embedding |
| `scripts/build_eruption_ground_truth.py` | Build the 62-event validation set from OpenAlex |
| `scripts/match_eruptions.py` | Match patent clusters to eruption events; measure lead time |
| `scripts/eval_test.py` | One-time locked test-set evaluation |
| `scripts/add_patent_ids.py` | Enrich cluster samples with patent IDs (links to Google Patents) |
| `scripts/add_assignees.py` | Add top filing organisations per cluster |
| `scripts/build_patent_db.py` | Build SQLite lookup for full-text patent search in dashboard |
| `scripts/download_patents.sh` | Re-fetch `g_patent.tsv.zip` from PatentsView S3 |
| `dashboard/app.py` | Dash/Plotly dashboard |
| `data/raw/patents/g_patent.tsv.zip` | 9.4M USPTO patents 1976–2025 (gitignored, 222MB) |
| `data/processed/validation/eruption_ground_truth.csv` | 62 objective eruption events |
| `data/processed/clusters/` | Cluster files, trained models, leaderboards |

---

## Data split

| Split | Period | Role |
|---|---|---|
| Train | 2005 Q1 – 2017 Q4 | Model fitting |
| Val | 2018 Q1 – 2020 Q4 | Hyperparameter tuning |
| **Test** | **2021 Q1 – 2023 Q4** | **Locked — evaluated once** |
| Future | 2024 Q1 → present | Leaderboard inference only |

The test gate in the notebook raises `RuntimeError` unless explicitly unlocked.

---

*Assisted by [Claude Code](https://claude.ai/code)*
