# Patent Emerging Radar — Project Context for Claude

## What this project does

Discovers and forecasts emerging scientific research topics from USPTO patent filings,
without any pre-curated topic list. Patents precede papers by 1–3 years — a cluster of
semantically-similar patents growing fast today is a leading signal of the next CRISPR or LLMs.

Output: a ranked leaderboard — "these patent clusters will explode in the next 2 years."

## CURRENT STATUS

### Kaggle embedding job — was running when session ended
- Notebook: `notebooks/kaggle_embed.ipynb`
- Running via: Save Version → Save & Run All
- Dataset on Kaggle: `/kaggle/input/datasets/peannut/patents1/g_patent.tsv`
- File only covers **2002–2025** (not 1996 — the user's Kaggle upload was partial)
- Total runtime: ~2–2.5 hours on T4 GPU
- When done: Output tab → download `embeddings.zip` (~6–7GB)

### When Kaggle finishes — do this:
```bash
cd /home/jin/Documents/GITHUB/patent-emerging-radar/data/processed/embeddings
unzip ~/Downloads/embeddings.zip
# You'll have 2002.npz through 2025.npz

# Optional: fill in 1996–2001 from local file (~30-45 min on CPU)
cd /home/jin/Documents/GITHUB/patent-emerging-radar
python scripts/embed_patents.py --years 1996 2001 --batch-size 256
```

### Then open the main notebook:
```bash
jupyter notebook notebooks/patent_emerging_radar.ipynb
```
Run Parts 0–1, skip Part 2, run the verify cell, continue from Part 3.

---

## Pipeline

```
USPTO patent titles (g_patent.tsv.zip, 9.4M patents 1976–2025 locally)
    ↓
[Part 2] sentence-transformers (all-MiniLM-L6-v2) → 384-d embeddings, one .npz per year
    ↓
[Part 3] UMAP (384d→10d) + HDBSCAN rolling 2-yr windows → patent clusters
         Cross-window stitching by centroid cosine similarity (threshold 0.85)
    ↓
[Part 4] 18 engineered features per cluster per quarter
         (log_count, count_share, velocity, acceleration, jerk,
          mom_4q, mom_8q, roll_mean/std 4q/8q, above_trend, macd,
          z_score, global_rank_pctl, log_cumsum, sin_q, cos_q)
    ↓
[Parts 5–6] GRU + LSTM — 8Q lookback → 8Q forecast (2-year horizon)
            Loss: Huber(delta=1.0). Primary metric: Spearman rank correlation.
    ↓
[Parts 7–8] Val set analysis. Retrospective: patent cluster lead time vs
            publication eruption. Nobel cross-check.
    ↓
[Part 9] 🔒 TEST SET — RUN_TEST_SET = False. Raises RuntimeError.
         DO NOT unlock without the user explicitly saying to.
    ↓
[Part 10] Leaderboard: top clusters predicted to erupt in next 2 years
```

---

## Train / Val / Test split (anchor-point based, no leakage)

| Split | Anchor range | Notes |
|---|---|---|
| Train | 2005 Q1 → 2017 Q4 | Fit weights |
| Val | 2018 Q1 → 2020 Q4 | All tuning here |
| Test | 2021 Q1 → 2023 Q4 | **LOCKED** |
| Future | 2024 Q1 → present | Leaderboard only |

---

## Validation (critical — read this)

**Do NOT use `data/processed/validation/panel.csv`** for validation.
It has selection bias — 37 topics chosen because they already blew up.

**Use `eruption_ground_truth.csv` instead:**
- 62 objective eruption events from 4,516 OpenAlex topics, no hand-picking
- Type A (emerging): <500 papers/yr → 5× growth in 5 years
- Type B (acceleration): <10k papers/yr → 4× growth in 5 years
- Nobel cross-check: T11289 single-cell (Nobel 2024) confirmed at 4.5× 2014→2019

Lead time metric: quarters between patent cluster takeoff (velocity ≥ 0.3) and OpenAlex eruption.

---

## Key files

| File | Purpose | Status |
|---|---|---|
| `notebooks/patent_emerging_radar.ipynb` | Main notebook (59 cells, Parts 0–10) | Ready to run after embeddings arrive |
| `notebooks/kaggle_embed.ipynb` | Kaggle GPU embedding job (16 cells) | Was running |
| `scripts/embed_patents.py` | Local embed. Default `--years 1996 2024 --batch-size 64` | Ready |
| `scripts/build_eruption_ground_truth.py` | Standalone eruption detection | Done |
| `scripts/download_patents.sh` | Re-fetch g_patent.tsv.zip from PatentsView S3 | Ready |
| `src/concepts.py` | 37 ground-truth topics for retrospective matching | Done |
| `data/raw/patents/g_patent.tsv.zip` | 222MB, **gitignored**, 1976–2025, 9.4M patents | Present locally |
| `data/raw/openalex/topics_cache.json` | 4,516 topics × quarterly counts 2000–2025 | Present |
| `data/processed/validation/eruption_ground_truth.csv` | 62 eruption events | Built |
| `data/processed/embeddings/` | .npz files from Kaggle go here | Pending |
| `BRAINSTORM.md` | Full project notes and business case | Present |

---

## Patent data facts

- Local `g_patent.tsv.zip`: 1976–2025, 9.4M total rows (confirmed by full scan)
- Kaggle upload (`g_patent.tsv`): 2002–2025 only (user's file was partial)
- Recommended embed start: 1996 (internet/genomics era, modern language quality)
- Pre-1990: limited value — different technology landscape, older language
- The `stream_patents()` filter is fully parameterised via `--years`

---

## Gotchas and decisions made

- **Per-year CSV zips (20XX.csv.zip)**: metadata/assignee files, NO patent titles. Cannot be used for embedding. Only g_patent.tsv.zip has titles.
- **Kaggle pip warnings**: dependency conflicts from Kaggle's own GPU libraries (dask-cuda, cuml etc.) — completely harmless, sentence-transformers doesn't use them
- **Kaggle output**: only appears after "Save Version → Save & Run All" (committed run), not after interactive "Run All"
- **Zip strategy**: ZIP_STORED + delete originals one at a time to stay under 19.5GB Kaggle limit
- **Eruption ground truth Nobel anchors**: CRISPR/gravitational waves/immunotherapy grew gradually (2-3×) not sharply enough to hit eruption thresholds — they serve as qualitative checks, not formal validation events. T11289 (single-cell) IS in the formal list.
- **Test gate**: RuntimeError, not just a print warning. User must set RUN_TEST_SET = True explicitly.

---

## To get back up to speed

1. Read this file
2. Check if Kaggle job finished: kaggle.com/code → your notebook → Output tab
3. If embeddings downloaded and extracted: open main notebook, run from Part 3
4. If still waiting: check Kaggle status, maybe fill in 1996–2001 locally

## Related project

`../scientific-trend-forecaster/` — sibling project, 37 hand-curated topics, working GRU/LSTM.
OpenAlex cache and reference files were copied from there for this project.
