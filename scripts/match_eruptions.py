"""
match_eruptions.py — retrospective validation

Matches the 62 known eruption events (eruption_ground_truth.csv) to the
closest patent cluster centroid, then measures how many quarters the patent
cluster took off BEFORE the publication eruption.

Run after unzipping pipeline_output.zip:
    python scripts/match_eruptions.py

Output: data/processed/clusters/retrospective_matches.csv
"""

import pathlib, pickle
import numpy as np
import pandas as pd

BASE  = pathlib.Path(__file__).parent.parent
CDIR  = BASE / 'data' / 'processed' / 'clusters'
VDIR  = BASE / 'data' / 'processed' / 'validation'
OUT   = CDIR / 'retrospective_matches.csv'

TAKEOFF_VELOCITY = 0.3   # log-count velocity threshold for "takeoff"
MIN_SIM          = 0.25  # min cosine similarity to call a match confident


def embed_texts(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        'sentence-transformers', '-q'], check=False)
        from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (n, d)  B: (m, d), both L2-normalised
    return A @ B.T


def find_takeoff_quarter(series: pd.Series, threshold: float) -> str | None:
    """First quarter where velocity >= threshold, using 2-quarter rolling mean."""
    vel = series.diff().fillna(0).rolling(2, min_periods=1).mean()
    hits = vel[vel >= threshold]
    return str(hits.index[0]) if not hits.empty else None


def main():
    # ── Load centroids ────────────────────────────────────────────────────────
    centroid_path = CDIR / 'cluster_centroids.pkl'
    if not centroid_path.exists():
        raise FileNotFoundError(
            f'{centroid_path} not found.\n'
            'Unzip pipeline_output.zip into data/processed/ first.'
        )
    with open(centroid_path, 'rb') as f:
        centroids = pickle.load(f)

    labels_df = pd.read_csv(CDIR / 'cluster_labels.csv')
    labels    = dict(zip(labels_df['cluster_id'], labels_df['auto_label']))

    cids  = sorted(centroids.keys())
    C_mat = np.array([centroids[i] for i in cids], dtype=np.float32)
    # Normalise (centroids may already be normalised, but be safe)
    norms = np.linalg.norm(C_mat, axis=1, keepdims=True) + 1e-9
    C_mat = C_mat / norms

    # ── Load eruption ground truth ────────────────────────────────────────────
    gt = pd.read_csv(VDIR / 'eruption_ground_truth.csv')
    print(f'Loaded {len(gt)} eruption events.')

    # ── Embed topic names ─────────────────────────────────────────────────────
    print('Embedding eruption topic names …')
    topic_embs = embed_texts(gt['topic_name'].tolist())  # already normalised

    # ── Cosine similarity: topics × clusters ──────────────────────────────────
    sim = cosine_sim_matrix(topic_embs, C_mat)  # (62, n_clusters)
    best_idx  = sim.argmax(axis=1)
    best_sim  = sim.max(axis=1)

    # ── Load panel for velocity / takeoff ─────────────────────────────────────
    panel = pd.read_csv(CDIR / 'cluster_panel.csv')
    panel['log_count'] = np.log1p(panel['count'])
    panel['date']      = pd.to_datetime(panel['date'])
    panel = panel.sort_values(['cluster_id', 'date'])

    # ── Build results ──────────────────────────────────────────────────────────
    rows = []
    for i, ev in gt.iterrows():
        cid      = cids[best_idx[i]]
        sim_val  = float(best_sim[i])
        auto_lbl = labels.get(cid, f'cluster_{cid}')
        confident = sim_val >= MIN_SIM

        # Eruption quarter: first quarter of eruption_year
        eruption_q = f'{int(ev["eruption_year"])}-01-01'

        # Cluster takeoff: first quarter where velocity smoothed >= threshold
        sub = panel[panel['cluster_id'] == cid].set_index('date')['log_count']
        takeoff_q = None
        if not sub.empty:
            # Look for takeoff before the eruption year
            pre = sub[sub.index < eruption_q]
            takeoff_q = find_takeoff_quarter(pre, TAKEOFF_VELOCITY)

        # Lead time in quarters
        lead_q = None
        if takeoff_q is not None:
            td = pd.Timestamp(eruption_q) - pd.Timestamp(takeoff_q)
            lead_q = int(round(td.days / 91.25))

        rows.append({
            'topic_id':       ev['topic_id'],
            'topic_name':     ev['topic_name'],
            'domain':         ev['domain'],
            'eruption_type':  ev['eruption_type'],
            'eruption_year':  int(ev['eruption_year']),
            'growth_x':       float(ev['growth_x']),
            'is_nobel':       bool(ev['is_nobel']),
            'cluster_id':     cid,
            'auto_label':     auto_lbl,
            'cosine_sim':     round(sim_val, 4),
            'confident':      confident,
            'cluster_takeoff': takeoff_q,
            'eruption_q':     eruption_q,
            'lead_time_q':    lead_q,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    conf = df[df['confident']]
    with_lead = conf[conf['lead_time_q'].notna() & (conf['lead_time_q'] > 0)]

    print(f'\nResults saved → {OUT}')
    print(f'Total eruption events  : {len(df)}')
    print(f'Confident matches (≥{MIN_SIM}) : {len(conf)}')
    print(f'With positive lead time: {len(with_lead)}')
    if not with_lead.empty:
        med = with_lead['lead_time_q'].median()
        mx  = with_lead['lead_time_q'].max()
        print(f'Median lead time       : {med:.0f} quarters ({med/4:.1f} yrs)')
        print(f'Max lead time          : {mx:.0f} quarters ({mx/4:.1f} yrs)')

    print('\nTop 10 confident matches:')
    top = (conf.sort_values('cosine_sim', ascending=False)
               .head(10)[['topic_name', 'auto_label', 'cosine_sim',
                           'cluster_takeoff', 'eruption_year', 'lead_time_q']])
    print(top.to_string(index=False))


if __name__ == '__main__':
    main()
