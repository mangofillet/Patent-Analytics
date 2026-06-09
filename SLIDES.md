# Patent Emerging Radar — Presentation Slides

---

## Slide 1 — The Big Idea

- **Patents lead publications by 1–3 years** — companies file before academics publish
- A cluster of similar patents **accelerating today** is a leading signal of the next CRISPR or LLM
- We forecast **which technology areas will explode next** — with no hand-picked topic list
- Output: a ranked leaderboard of emerging tech, straight from raw USPTO filings

---

## Slide 2 — The Data

- **5.6M USPTO patents**, 2002–2025, every quarter
- Fully **unsupervised** — no labels, no look-ahead, no cherry-picking
- Discovered **2,006 technology clusters** automatically
- 96 quarters of history per cluster to learn growth dynamics

---

## Slide 3 — How It Works (Pipeline)

- **Embed** — every patent title → 384-dim vector (all-MiniLM-L6-v2)
- **Cluster** — UMAP + HDBSCAN on rolling 2-year windows
- **Stitch** — link clusters across time by centroid similarity → stable topic IDs
- **Features** — 18 time-series signals per cluster (velocity, acceleration, momentum, MACD…)
- **Forecast** — neural net predicts 2- and 3-year growth → ranked leaderboard

---

## Slide 4 — The Model

- Two architectures trained head-to-head: **GRU vs LSTM**
- Input: 8 quarters × 18 features → RNN(64) → RNN(32) → growth prediction
- Target: **growth delta** = log_count[t+H] − log_count[t]
- **LSTM won both horizons** — selected as the production model
- Honest evaluation: ranked by **Spearman correlation**, test set kept locked until the end

---

## Slide 5 — Does It Actually Work? (Results)

- **Test ρ = 0.42 (2yr) / 0.47 (3yr)** — ranks future winners well above chance
- **No overfitting** — test score ≈ validation score (even slightly higher)
- **Beats the baseline by +0.55** — naive momentum is *negative* (−0.13), it mis-ranks growth
- **Robust signal** — 2yr and 3yr models independently flag 11 of the same top-20 clusters
- The model is **selective**: only 56% of clusters forecast to grow, not blindly bullish

---

## Slide 6 — The Dashboard (Live Demo)

- **3D Technology Galaxy** — every cluster a star, sized by volume, colored by forecast growth
- **Click any node** → keywords, trend chart, real patent titles
- **Search + domain filters** — zoom into AI, biotech, energy, etc.
- **INSIGHTS / METRICS / PIPELINE tabs** — expand to full-screen graphical explainers
- Everything traces back to real patents — fully transparent, no black box

---

## Slide 7 — Why It Matters

- **R&D strategy** — see where a field is heading before it's obvious
- **Investment** — spot emerging tech 1–3 years ahead of the publication wave
- **Policy / funding** — allocate to genuinely accelerating areas, backed by evidence
- A repeatable, unsupervised radar — re-runs on fresh patents anytime

---

### One-liner for the title slide
> **Patent Emerging Radar** — forecasting the next breakthrough technologies from 5.6M USPTO patents, fully unsupervised, validated on a locked test set.
