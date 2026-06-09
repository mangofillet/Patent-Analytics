import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import dash
from dash import dcc, html, Input, Output, State, Patch, ctx, ALL
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from data_loader import load_all

BASE = pathlib.Path(__file__).parent.parent
CDIR = BASE / 'data' / 'processed' / 'clusters'

# ── Startup ────────────────────────────────────────────────────────────────────
print('[app] Loading data …')
D = load_all()
print('[app] Ready.')

# ── Palette ────────────────────────────────────────────────────────────────────
BG     = '#070b14'
BORDER = 'rgba(255,255,255,0.07)'
NEON   = '#00d4ff'
VIOLET = '#7c3aed'
AMBER  = '#f59e0b'
TEXT   = '#e2e8f0'
MUTED  = '#475569'

# Diverging: strong-decline RED → mild-decline ORANGE → neutral slate → growth CYAN → strong-growth MINT
GROWTH_CS = [[0.0,'#ef4444'],[0.22,'#f59e0b'],[0.5,'#1e2740'],[0.78,'#22d3ee'],[1.0,'#5eead4']]
AGE_CS    = [[0.0,'#6366f1'],[0.5,'#0ea5e9'],[1.0,'#f59e0b']]
VEL_CS    = [[0.0,'#ef4444'],[0.5,'#1e2740'],[1.0,'#34d399']]

# ── Precompute cluster birth quarters from panel ────────────────────────────────
def _birth_years(panel):
    if panel is None:
        return {}
    # birth = first quarter the cluster actually had patents (count>0).
    # The panel carries zero-count rows back to 2002 for every cluster, so a plain
    # min(date) would report 2002 for everything — use the first *active* quarter.
    active = panel[panel['count'] > 0]
    first = active.groupby('cluster_id')['date'].min()
    return {int(k): pd.to_datetime(v).year for k, v in first.items()}

BIRTH   = _birth_years(D['panel'])
DOMAINS = D.get('domains', {})

# Sorted unique domain list for chip buttons
ALL_DOMAINS = sorted({v for v in DOMAINS.values() if v != 'Other'}) + ['Other']

# ── Known validation metrics from the training run (kaggle_train.ipynb) ──────────
VAL_METRICS = {
    '2yr': {
        'GRU':             {'spearman': 0.3900, 'rmse': 1.1669, 'mae': 0.7846},
        'LSTM':            {'spearman': 0.4143, 'rmse': 1.1580, 'mae': 0.7720},
        'Linear trend':    {'spearman': -0.1212, 'rmse': 2.1174, 'mae': 1.4195},
        'Persistence (0)': {'spearman': 0.0,    'rmse': 1.3183, 'mae': 0.8242},
    },
    '3yr': {
        'GRU':             {'spearman': 0.4461, 'rmse': 1.2745, 'mae': 0.8590},
        'LSTM':            {'spearman': 0.4556, 'rmse': 1.2807, 'mae': 0.8680},
        'Linear trend':    {'spearman': -0.1376, 'rmse': 2.8782, 'mae': 1.9457},
        'Persistence (0)': {'spearman': 0.0,    'rmse': 1.5180, 'mae': 0.9725},
    },
}
BEST_MODEL = {'2yr': 'LSTM', '3yr': 'LSTM'}  # LSTM won both horizons on val

# Where the skill lives — from scripts/analyze_skill.py on the 2yr validation set.
# Directional precision at the extremes vs the 37% base rate.
SKILL_REGION = {
    'base_rate_grow': 0.37,
    'top10_grew':     0.74,   # of top-10% predicted, fraction that actually grew
    'bottom10_decl':  0.75,   # of bottom-10% predicted, fraction that actually declined
    'rho_top':    0.487,      # within-band Spearman (range-restricted)
    'rho_middle': 0.303,
    'rho_bottom': 0.090,
}

def _load_test_metrics():
    import json
    p = CDIR / 'test_metrics.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None
TEST_METRICS = _load_test_metrics()


# ── Conviction score + acceleration (computed once at startup) ───────────────────
def _compute_conviction():
    """
    Blend signals into a 0–100 'conviction' score per cluster — how much to TRUST
    a forecast, not just how big it is. Combines (percentile-ranked):
      50%  forecast growth   — mean of 2yr & 3yr (they correlate ρ≈0.9, so treated
                               as ONE signal to avoid double-counting; agreement is
                               the norm, captured by averaging not double-weighting)
      25%  patent volume     — established signal, less small-sample noise
      25%  recent accel.     — momentum still building (2nd derivative of log-count)
    NOTE: heuristic, hand-set weights — a triage aid, not a calibrated probability.
    Also returns per-cluster acceleration so we can flag 'accelerating' clusters.
    """
    panel, lb2, lb3 = D['panel'], D['lb2'], D['lb3']
    if lb2 is None or panel is None:
        return {}, {}, None

    df = lb2[['cluster_id', 'auto_label', 'predicted_growth', 'last_log_count']].copy()
    df = df.rename(columns={'predicted_growth': 'growth_2yr', 'last_log_count': 'volume'})
    if lb3 is not None:
        g3 = lb3[['cluster_id', 'predicted_growth']].rename(columns={'predicted_growth': 'growth_3yr'})
        df = df.merge(g3, on='cluster_id', how='left')
    else:
        df['growth_3yr'] = df['growth_2yr']

    # recent acceleration = mean of last 4 quarters of 2nd-difference of log_count
    accel = (panel.sort_values('date').groupby('cluster_id')['log_count']
             .apply(lambda s: s.diff().diff().tail(4).mean()))
    df['accel'] = df['cluster_id'].map(accel).fillna(0.0)

    # 2yr & 3yr growth are ~0.9 correlated → average into a single growth signal
    df['growth_mean'] = df[['growth_2yr', 'growth_3yr']].mean(axis=1)

    def pct(col):
        return df[col].rank(pct=True)
    df['conviction'] = (0.50 * pct('growth_mean')
                        + 0.25 * pct('volume') + 0.25 * pct('accel')) * 100
    df['conviction'] = df['conviction'].round(1)

    conv = dict(zip(df['cluster_id'].astype(int), df['conviction']))
    acc  = dict(zip(df['cluster_id'].astype(int), df['accel']))
    # accelerating threshold = top quartile of positive acceleration
    return conv, acc, df

CONVICTION, ACCEL, CONV_DF = _compute_conviction()
ACCEL_THRESH = (np.nanpercentile([v for v in ACCEL.values() if v > 0], 75)
                if ACCEL and any(v > 0 for v in ACCEL.values()) else 0.05)

def is_accelerating(cid):
    return ACCEL.get(int(cid), 0.0) >= ACCEL_THRESH


# ── Company-level view: invert clusters → filers (computed once at startup) ───────
def _compute_companies():
    """
    Invert cluster→filer into a company-level investor table.
    Per company (across the clusters its sample patents appear in):
      n_clusters, n_patents, exposure (Σ count×max(0,g2)),
      concentration (exposure / n_patents — patent-weighted avg positive growth),
      top_theme (label of their highest count×g2 cluster), accelerating flag.
    """
    from collections import defaultdict
    assignees = D.get('assignees', {})
    lb2, lb3 = D['lb2'], D['lb3']
    if not assignees or lb2 is None:
        return None, {}
    g2 = dict(zip(lb2['cluster_id'].astype(int), lb2['predicted_growth']))
    lbl = D['labels']

    agg = defaultdict(lambda: {'n_clusters': 0, 'n_patents': 0, 'exposure': 0.0,
                               'best_score': -1e9, 'top_theme': '', 'accel': False})
    org_clusters = defaultdict(list)   # org -> [cluster_id, ...]
    for cid, orgs in assignees.items():
        cid = int(cid)
        gr  = float(g2.get(cid, 0.0))
        pos = max(0.0, gr)
        theme = str(lbl.get(cid, f'Cluster {cid}')).split(',')[0].strip()
        acc = is_accelerating(cid) and gr > 0
        for org, count in orgs:
            a = agg[org]
            a['n_clusters'] += 1
            a['n_patents']  += int(count)
            a['exposure']   += count * pos
            org_clusters[org].append(cid)
            score = count * pos
            if score > a['best_score']:
                a['best_score'] = score
                a['top_theme']  = theme
            if acc:
                a['accel'] = True

    rows = []
    for org, a in agg.items():
        npat = a['n_patents']
        rows.append({
            'org': org, 'n_clusters': a['n_clusters'], 'n_patents': npat,
            'exposure': round(a['exposure'], 2),
            'concentration': round(a['exposure'] / npat, 3) if npat else 0.0,
            'top_theme': a['top_theme'], 'accelerating': a['accel'],
        })
    df = pd.DataFrame(rows)
    return df, {o: sorted(set(c)) for o, c in org_clusters.items()}

COMPANIES, COMPANY_CLUSTERS = _compute_companies() if D.get('assignees') and D['lb2'] is not None else (None, {})


# ── Key analytics (computed once at startup) ─────────────────────────────────────
def _compute_analytics():
    A = {}
    panel, lab = D['panel'], D['labels']
    lb2, lb3 = D['lb2'], D['lb3']
    if lb2 is not None:
        s2 = lb2.sort_values('predicted_growth', ascending=False)
        A['top2_id']   = int(s2.iloc[0]['cluster_id'])
        A['top2_lbl']  = str(s2.iloc[0]['auto_label'])
        A['top2_grow'] = float(s2.iloc[0]['predicted_growth'])
        A['pos2']      = int((lb2['predicted_growth'] > 0).sum())
        A['n_clusters'] = len(lb2)
        A['med2']      = float(lb2['predicted_growth'].median())
        if lb3 is not None:
            s3 = lb3.sort_values('predicted_growth', ascending=False)
            A['top3_id']  = int(s3.iloc[0]['cluster_id'])
            A['top3_lbl'] = str(s3.iloc[0]['auto_label'])
            A['top3_grow']= float(s3.iloc[0]['predicted_growth'])
            t20_2 = set(s2.head(20).cluster_id)
            t20_3 = set(s3.head(20).cluster_id)
            A['overlap'] = len(t20_2 & t20_3)
    if panel is not None:
        A['patents'] = int(panel['count'].sum())
        A['quarters'] = int(panel['date'].nunique())
        vel = (panel.sort_values('date').groupby('cluster_id')['log_count']
               .apply(lambda s: s.diff().tail(4).mean()))
        vid = int(vel.sort_values(ascending=False).index[0])
        A['vel_id'], A['vel_lbl'], A['vel_val'] = vid, str(lab.get(vid, '')), float(vel.max())
        recent = panel.sort_values('date').groupby('cluster_id')['count'].last()
        bid = int(recent.sort_values(ascending=False).index[0])
        A['big_id'], A['big_lbl'], A['big_val'] = bid, str(lab.get(bid, '')), int(recent.max())
    return A

ANALYTICS = _compute_analytics()


def _analytics_rows():
    """Returns list of (metric, value, why) — the key analytics, ranked by importance."""
    A = ANALYTICS
    rows = []
    if TEST_METRICS:
        rows.append(('Forecast skill (test ρ)',
                     f"{TEST_METRICS['2yr']['test']['spearman']:+.2f} / {TEST_METRICS['3yr']['test']['spearman']:+.2f}  (2yr / 3yr)",
                     'Held-out Spearman. The single number that says the model works — it ranks future winners better than chance.'))
        rows.append(('Generalization gap',
                     f"≈ 0  (test ≥ val)",
                     'Test ρ matches/exceeds validation → no overfitting. The model learned a real pattern, not noise.'))
        rows.append(('Lift over baseline',
                     '+0.55 ρ vs linear trend',
                     'Linear momentum scores −0.13 (mis-ranks growth). The model adds genuine, non-obvious signal.'))
    if 'top2_lbl' in A:
        rows.append(('#1 emerging cluster (2yr)',
                     f"{A['top2_lbl'].split(',')[0]}  (+{A['top2_grow']:.2f} log)",
                     'The top actionable call — highest predicted 2-year growth across all clusters.'))
    if 'overlap' in A:
        rows.append(('Signal stability',
                     f"{A['overlap']}/20 top clusters agree across horizons",
                     '2yr and 3yr models independently flag the same clusters → the signal is robust, not horizon-specific.'))
    if 'vel_lbl' in A:
        rows.append(('Fastest accelerating now',
                     f"{A['vel_lbl'].split(',')[0]}  ({A['vel_val']:+.2f}/Q)",
                     'Steepest recent patent velocity — the rawest leading indicator before the model smooths it.'))
    if 'pos2' in A:
        rows.append(('Breadth of growth',
                     f"{A['pos2']:,} of {A['n_clusters']:,} clusters rising",
                     f"Only {100*A['pos2']//A['n_clusters']}% are forecast to grow — the model is selective, not bullish on everything."))
    if 'patents' in A:
        rows.append(('Corpus scale',
                     f"{A['patents']/1e6:.1f}M patents · {A['n_clusters']:,} clusters · {A['quarters']}Q",
                     'Unsupervised over the full 2002–2025 USPTO corpus — no hand-picked topics, no look-ahead.'))
    return rows

# ── Filter helpers ─────────────────────────────────────────────────────────────
# Reverse index: patent number → cluster (for the sample patents we have IDs for)
def _build_patent_index():
    idx = {}
    for cid, items in (D.get('titles_ids') or {}).items():
        for pid, _ in items:
            if pid:
                idx[str(pid)] = int(cid)
    return idx
PATENT_TO_CLUSTER = _build_patent_index()

# Full-coverage fallback: embed a patent's title → nearest cluster centroid.
PATENT_DB = BASE / 'data' / 'processed' / 'patent_titles.db'
# centroid matrix restricted to labeled clusters, L2-normalised for cosine
_CENTROID_IDS = sorted(int(c) for c in D['centroids'] if int(c) in D['labels'])
if _CENTROID_IDS:
    _CMAT = np.array([D['centroids'][c] for c in _CENTROID_IDS], dtype=np.float32)
    _CMAT = _CMAT / (np.linalg.norm(_CMAT, axis=1, keepdims=True) + 1e-9)
else:
    _CMAT = None
_EMBED_MODEL = None  # lazy-loaded on first fallback query


def _patent_title_db(num: str):
    """Look up a patent title by number from the SQLite index (any of 9.4M patents)."""
    import sqlite3
    if not PATENT_DB.exists():
        return None
    try:
        con = sqlite3.connect(f'file:{PATENT_DB}?mode=ro', uri=True)
        row = con.execute('SELECT title FROM patents WHERE id=?', (num,)).fetchone()
        con.close()
        return row[0] if row else None
    except Exception:
        return None


def _embed_match_cluster(title: str):
    """Embed a title and return the nearest cluster id by cosine, or None."""
    global _EMBED_MODEL
    if _CMAT is None or not title:
        return None
    try:
        if _EMBED_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        v = _EMBED_MODEL.encode([title], normalize_embeddings=True)[0].astype(np.float32)
        sims = _CMAT @ v
        return int(_CENTROID_IDS[int(sims.argmax())])
    except Exception:
        return None


def _patent_lookup(search: str):
    """If `search` looks like a US patent number, return (is_patent, cluster_id, number).
    Tries the sampled index first, then the embed-match fallback over all 9.4M patents."""
    import re
    compact = re.sub(r'[^0-9a-z]', '', (search or '').lower())
    m = re.fullmatch(r'(?:us)?(\d{6,9})[a-z]?\d?', compact)
    if not m:
        return False, None, None
    num = m.group(1)
    cid = PATENT_TO_CLUSTER.get(num) or PATENT_TO_CLUSTER.get(num.lstrip('0'))
    if cid is None:
        # fallback: pull the title from the DB, embed it, match nearest centroid
        title = _patent_title_db(num) or _patent_title_db(num.lstrip('0'))
        if title:
            cid = _embed_match_cluster(title)
    return True, cid, num


# Pre-warm the embedding model in the background so the first patent-fallback
# search is instant instead of paying ~10s of model-load latency.
if PATENT_DB.exists() and _CMAT is not None:
    import threading
    threading.Thread(target=lambda: _embed_match_cluster('warmup'), daemon=True).start()


def _filter_ids(search: str, domain: str) -> set | None:
    """Return set of matching cluster IDs, or None if no filter is active."""
    search = (search or '').strip().lower()
    active = bool(search) or bool(domain)
    if not active:
        return None

    # Patent-number search: "US7036143" / "7036143" / "7,036,143"
    if search:
        is_patent, cid, _num = _patent_lookup(search)
        if is_patent:
            return {int(cid)} if cid is not None else set()

    labels  = D['labels']
    titles  = D['titles']
    domains = DOMAINS
    result  = set()

    for cid, lbl in labels.items():
        # domain filter
        if domain and domains.get(cid) != domain:
            continue
        # keyword search — check label + sample titles
        if search:
            hay = lbl.lower()
            hay += ' ' + ' '.join(str(t).lower() for t in titles.get(cid, [])[:5])
            if search not in hay:
                continue
        result.add(int(cid))

    return result


# ── Galaxy helpers ─────────────────────────────────────────────────────────────
def _lb(horizon):
    return {'2yr': D['lb2'], '3yr': D['lb3'], '5yr': D['lb5']}.get(horizon)

def _gcol(horizon):
    return 'extrapolated_growth_5yr' if horizon == '5yr' else 'predicted_growth'

def _node_props(horizon, color_mode, time_year=None, filter_ids=None):
    coords = D['coords3d']
    n = len(coords)
    if n == 0:
        return [], [], []

    lb   = _lb(horizon)
    gcol = _gcol(horizon)
    cids = [int(c) for c in coords['cluster_id']]

    # sizes: from leaderboard if available, else from panel recent count
    if lb is not None:
        vol = dict(zip(lb['cluster_id'].astype(int), lb['last_log_count']))
        sizes = [vol.get(c, 0) for c in cids]
    elif D['panel'] is not None:
        recent_count = (D['panel'].sort_values('date')
                        .groupby('cluster_id')['log_count'].last())
        sizes = [float(recent_count.get(c, 0.0)) for c in cids]
    else:
        sizes = [8.0] * n
    mx = max(sizes) or 1
    sizes = [max(3, min(18, s / mx * 15 + 3)) for s in sizes]

    # colors
    if color_mode == 'growth':
        if lb is not None:
            gmap = dict(zip(lb['cluster_id'].astype(int), lb[gcol]))
            colors = [gmap.get(c, 0.0) for c in cids]
        else:
            # fall back to velocity when no forecast data yet
            if D['panel'] is not None:
                recent = (D['panel'].sort_values('date')
                          .groupby('cluster_id')['velocity'].last())
                colors = [float(recent.get(c, 0.0)) for c in cids]
            else:
                colors = [0.0] * n
        cs = GROWTH_CS if lb is not None else VEL_CS
    elif color_mode == 'birth':
        colors = [float(BIRTH.get(c, 2010)) for c in cids]
        cs = AGE_CS
    else:  # velocity
        if D['panel'] is not None:
            recent = (D['panel'].sort_values('date')
                      .groupby('cluster_id')['velocity'].last())
            colors = [float(recent.get(c, 0.0)) for c in cids]
        else:
            colors = [0.0] * n
        cs = VEL_CS

    # Encode filter/time state in size: unmatched nodes shrink to tiny dots.
    # When a search/domain filter is active, matched dots use a UNIFORM size so the
    # view reads as "here are the related clusters" (colour still shows growth) rather
    # than varying by volume. Time-only fading keeps the volume sizing.
    if filter_ids is not None or time_year is not None:
        final_sizes = []
        for c, s in zip(cids, sizes):
            in_time   = (time_year is None) or (BIRTH.get(c, 2002) <= time_year)
            in_filter = (filter_ids is None) or (c in filter_ids)
            if in_time and in_filter:
                final_sizes.append(9.0 if filter_ids is not None else s)
            else:
                final_sizes.append(1.2)
        sizes = final_sizes

    return sizes, colors, cs


def build_galaxy(horizon='2yr', color_mode='growth', time_year=None, filter_ids=None):
    coords = D['coords3d']
    labels = D['labels']

    if coords.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(bgcolor='rgba(0,0,0,0)'),
            annotations=[dict(
                text='No cluster data yet.<br><span style="font-size:12px;color:#475569">'
                     'Run the Kaggle pipeline, then unzip pipeline_output.zip into data/processed/</span>',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                font=dict(color=MUTED, size=15), align='center',
            )]
        )
        return fig

    sizes, colors, cs = _node_props(horizon, color_mode, time_year, filter_ids)
    cids   = [int(c) for c in coords['cluster_id']]
    node_labels = [labels.get(c, f'Cluster {c}') for c in cids]

    c_arr = np.array(colors, dtype=float)
    c_abs = max(abs(c_arr.min()), abs(c_arr.max()), 0.01)
    cmin, cmax = -c_abs, c_abs
    if color_mode in ('birth', 'velocity'):
        cmin, cmax = float(c_arr.min()), float(c_arr.max())

    colorbar_title = {'growth': 'Forecast growth<br>(Δ log-count)', 'birth': 'Birth year',
                      'velocity': 'Recent velocity<br>(Δ log/Q)'}[color_mode]

    fig = go.Figure(go.Scatter3d(
        x=coords['x'], y=coords['y'], z=coords['z'],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=cs,
            cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text=colorbar_title, font=dict(color=TEXT, size=10)),
                tickfont=dict(color=MUTED, size=9),
                thickness=6, len=0.45, x=1.01,
                bordercolor='rgba(0,0,0,0)', bgcolor='rgba(0,0,0,0)',
            ),
            opacity=0.88,
            line=dict(width=0),
        ),
        text=node_labels,
        customdata=cids,
        hovertemplate='<b>%{text}</b><br>%{marker.color:.3f}<br><i>click to explore</i><extra></extra>',
    ))

    ax = dict(showgrid=False, showticklabels=False, showline=False,
              zeroline=False, showspikes=False, title='',
              backgroundcolor='rgba(0,0,0,0)')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=60, t=0, b=0),
        showlegend=False,
        uirevision='galaxy-stable',
        scene=dict(bgcolor='rgba(0,0,0,0)', xaxis=ax, yaxis=ax, zaxis=ax,
                   camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))),
    )
    return fig


# ── Trend chart ────────────────────────────────────────────────────────────────
def build_trend(cid, horizon='2yr'):
    fig = go.Figure()
    panel = D['panel']

    if panel is not None:
        sub = panel[panel['cluster_id'] == cid].sort_values('date')
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub['date'], y=sub['log_count'], mode='lines',
                line=dict(color=NEON, width=1.8),
                name='Historical',
            ))

    lb = _lb(horizon)
    if lb is not None:
        gcol = _gcol(horizon)
        m = lb[lb['cluster_id'] == cid]
        if not m.empty:
            last_log  = float(m.iloc[0]['last_log_count'])
            delta     = float(m.iloc[0][gcol])
            last_date = pd.to_datetime(m.iloc[0]['last_date'])
            hq = {'2yr': 8, '3yr': 12, '5yr': 20}[horizon]
            fut  = pd.date_range(last_date, periods=hq + 1, freq='QS')
            futy = np.linspace(last_log, last_log + delta, hq + 1)
            lc   = NEON if delta > 0 else VIOLET
            fig.add_trace(go.Scatter(
                x=fut, y=futy, mode='lines',
                line=dict(color=lc, width=1.8, dash='dash'),
                name=f'{horizon} forecast', opacity=0.75,
            ))
            fig.add_vrect(x0=last_date, x1=fut[-1],
                          fillcolor=f'rgba(0,212,255,0.04)' if delta > 0
                          else 'rgba(124,58,237,0.04)', line_width=0)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT), showlegend=True,
        legend=dict(font=dict(color=MUTED, size=9), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=36, r=8, t=8, b=36),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color=MUTED,
                   tickfont=dict(size=9), showline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color=MUTED,
                   title=dict(text='log(1+count)', font=dict(size=9, color=MUTED)),
                   showline=False),
        hovermode='x unified',
    )
    return fig


# ── Leaderboard panel ──────────────────────────────────────────────────────────
def build_leaderboard(horizon='2yr', direction='growers'):
    lb = _lb(horizon)
    if lb is None:
        # No forecast yet — show top clusters by recent patent velocity from panel
        if D['panel'] is None:
            return html.Div([
                html.P('No data yet.', style={'color': MUTED, 'textAlign': 'center', 'marginTop': '48px', 'fontSize': '13px'}),
            ])
        recent = (D['panel'].sort_values('date')
                  .groupby('cluster_id')
                  .agg(velocity=('velocity', 'last'), last_count=('count', 'last'))
                  .reset_index()
                  .sort_values('velocity', ascending=False)
                  .head(20))
        mx  = float(recent['velocity'].abs().max()) or 1
        rows = []
        for i, row in enumerate(recent.itertuples(), 1):
            g   = float(row.velocity)
            w   = abs(g) / mx * 100
            bc  = NEON if g > 0 else VIOLET
            lbl = str(D['labels'].get(int(row.cluster_id), f'Cluster {row.cluster_id}'))
            rows.append(html.Div([
                html.Span(f'{i}', style={'color': '#334155', 'fontSize': '10px', 'width': '18px', 'flexShrink': '0', 'fontFamily': 'monospace'}),
                html.Span(lbl[:34] + ('…' if len(lbl) > 34 else ''), style={
                    'flex': '1', 'fontSize': '12px', 'color': TEXT,
                    'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap',
                }),
                html.Div(html.Div(style={
                    'height': '3px', 'width': f'{w:.0f}%',
                    'background': bc, 'borderRadius': '2px',
                    'boxShadow': f'0 0 5px {bc}55',
                }), style={'width': '70px', 'flexShrink': '0'}),
                html.Span(f'{g:+.3f}', style={
                    'color': bc, 'fontSize': '11px', 'width': '50px', 'textAlign': 'right',
                    'flexShrink': '0', 'fontFamily': 'Space Grotesk, monospace',
                }),
            ], id={'type': 'lb-row', 'index': int(row.cluster_id)},
               className='lb-row', n_clicks=0))
        return html.Div([
            html.Div('FASTEST GROWING — RECENT VELOCITY  (no forecast yet)',
                     style={'color': '#334155', 'fontSize': '9px', 'letterSpacing': '1.5px',
                            'padding': '8px 12px', 'borderBottom': f'1px solid {BORDER}'}),
            html.Div(rows, style={'overflowY': 'auto', 'height': 'calc(100% - 30px)'}),
        ], style={'height': '100%'})

    gcol = _gcol(horizon)
    if direction == 'decliners':
        top = lb.nsmallest(20, gcol)
    else:
        top = lb.nlargest(20, gcol)
    mx   = float(top[gcol].abs().max()) or 1

    rows = []
    for i, row in enumerate(top.itertuples(), 1):
        g   = float(getattr(row, gcol.replace('-','_')))
        w   = abs(g) / mx * 100
        bc  = NEON if g > 0 else VIOLET
        lbl = str(row.auto_label)
        accel = is_accelerating(row.cluster_id)
        rows.append(html.Div([
            html.Span(f'{i}', style={'color': '#334155', 'fontSize': '10px', 'width': '18px', 'flexShrink': '0', 'fontFamily': 'monospace'}),
            html.Span([
                *([html.Span('⚡', title='Acceleration still building',
                             style={'color': '#fbbf24', 'marginRight': '4px', 'fontSize': '10px'})] if accel else []),
                lbl[:32] + ('…' if len(lbl) > 32 else ''),
            ], style={
                'flex': '1', 'fontSize': '12px', 'color': TEXT,
                'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap',
            }),
            html.Div(html.Div(style={
                'height': '3px', 'width': f'{w:.0f}%',
                'background': bc, 'borderRadius': '2px',
                'boxShadow': f'0 0 5px {bc}55',
            }), style={'width': '70px', 'flexShrink': '0'}),
            html.Span(f'{g:+.3f}', style={
                'color': bc, 'fontSize': '11px', 'width': '50px', 'textAlign': 'right',
                'flexShrink': '0', 'fontFamily': 'Space Grotesk, monospace',
            }),
        ], id={'type': 'lb-row', 'index': int(row.cluster_id)},
           className='lb-row', n_clicks=0))

    return html.Div(rows, style={'overflowY': 'auto', 'height': '100%'})


# ── Cluster detail panel ───────────────────────────────────────────────────────
def build_detail(cid, horizon='2yr'):
    label = D['labels'].get(cid, f'Cluster {cid}')
    titles = D['titles'].get(cid, [])[:6]
    birth = BIRTH.get(cid)
    keywords = [k.strip() for k in label.split(',')]

    lb = _lb(horizon)
    growth_str = ''
    if lb is not None:
        gcol = _gcol(horizon)
        m = lb[lb['cluster_id'] == cid]
        if not m.empty:
            g = float(m.iloc[0][gcol])
            c = NEON if g > 0 else VIOLET
            growth_children = [html.Span(f'{g:+.3f}', style={
                'color': c, 'fontFamily': 'Space Grotesk, monospace',
                'fontSize': '22px', 'fontWeight': '700',
                'textShadow': f'0 0 14px {c}88'})]
            # v2 prediction interval (P10–P90, conformal) when available
            if 'growth_lo' in m.columns and 'growth_hi' in m.columns:
                lo, hi = float(m.iloc[0]['growth_lo']), float(m.iloc[0]['growth_hi'])
                growth_children.append(html.Span(
                    f'  80% CI [{lo:+.2f}, {hi:+.2f}]',
                    style={'color': MUTED, 'fontSize': '11px', 'fontFamily': 'Space Grotesk, monospace'}))
            growth_str = html.Span(growth_children)

    # Conviction gauge + acceleration badge
    conv = CONVICTION.get(int(cid))
    conv_block = None
    if conv is not None:
        cc = '#10b981' if conv >= 70 else (AMBER if conv >= 45 else VIOLET)
        conv_tip = ('Conviction (0–100): heuristic triage score — 50% forecast growth, '
                    '25% patent volume, 25% acceleration. Not a probability. '
                    '≥70 strong · 45–69 moderate · <45 speculative.')
        conv_block = html.Div([
            html.Div([
                html.Span('CONVICTION ⓘ', title=conv_tip,
                          style={'color': MUTED, 'fontSize': '9px', 'letterSpacing': '1.5px', 'cursor': 'help'}),
                html.Span(f'{conv:.0f}', style={'color': cc, 'fontFamily': 'Space Grotesk, monospace',
                          'fontSize': '20px', 'fontWeight': '700', 'marginLeft': '8px'}),
                html.Span('/100', style={'color': MUTED, 'fontSize': '11px'}),
                *([html.Span('⚡ ACCELERATING', title='Patent velocity is still rising (positive 2nd derivative) — momentum building, not yet peaked.',
                    style={
                    'color': '#fbbf24', 'fontSize': '9px', 'fontWeight': '700', 'letterSpacing': '1px',
                    'marginLeft': 'auto', 'border': '1px solid rgba(251,191,36,0.4)',
                    'borderRadius': '10px', 'padding': '2px 8px', 'cursor': 'help',
                    'boxShadow': '0 0 8px rgba(251,191,36,0.3)'})] if is_accelerating(cid) else []),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
            html.Div(html.Div(style={'height': '4px', 'width': f'{conv:.0f}%', 'background': cc,
                     'borderRadius': '2px', 'boxShadow': f'0 0 6px {cc}88'}),
                     style={'background': 'rgba(255,255,255,0.06)', 'borderRadius': '2px'}),
        ], style={'marginBottom': '12px'})

    chips = [html.Span(k, className='keyword-chip') for k in keywords if k]

    # Sample patents — with real patent numbers + Google Patents links when available
    titles_ids = D.get('titles_ids', {}).get(cid)
    if titles_ids:
        title_items = []
        for pid, t in titles_ids[:6]:
            disp = t[:80] + ('…' if len(t) > 80 else '')
            if pid:
                num = html.A(f'US{pid}', href=f'https://patents.google.com/patent/US{pid}',
                             target='_blank', className='patent-num')
            else:
                num = html.Span('—', style={'color': MUTED, 'fontSize': '10px'})
            title_items.append(html.Li([
                num,
                html.Span(disp, style={'color': '#64748b', 'fontSize': '11px', 'marginLeft': '8px'}),
            ], style={'marginBottom': '6px', 'display': 'flex', 'alignItems': 'baseline'}))
    else:
        title_items = [html.Li(t[:90]+'…' if len(t)>90 else t,
                               style={'color':'#64748b','fontSize':'11px','marginBottom':'4px'})
                       for t in titles]

    # Top filers — "who's filing", each expandable to its patents (numbers + links)
    filers_detail = D.get('filer_pats', {}).get(cid) or []
    filer_block = None
    if filers_detail:
        filer_rows = []
        for org, n, pats in filers_detail[:8]:
            pat_links = []
            for pid, t in pats:
                disp = (t[:62] + '…') if t and len(t) > 62 else (t or '')
                pat_links.append(html.Li([
                    html.A(f'US{pid}', href=f'https://patents.google.com/patent/US{pid}',
                           target='_blank', className='patent-num') if pid
                    else html.Span('—', style={'color': MUTED}),
                    html.Span(disp, style={'color': '#64748b', 'fontSize': '11px', 'marginLeft': '8px'}),
                ], style={'marginBottom': '5px', 'display': 'flex', 'alignItems': 'baseline'}))
            filer_rows.append(html.Details([
                html.Summary([
                    html.Span(org, style={'color': TEXT, 'fontWeight': '600'}),
                    html.Span(f'  {n} patent{"s" if n != 1 else ""}',
                              style={'color': NEON, 'fontSize': '9px', 'marginLeft': '6px'}),
                ], className='filer-summary'),
                html.Ul(pat_links, style={'listStyle': 'none', 'padding': '6px 0 4px 10px', 'margin': '0'}),
            ], className='filer-details'))
        filer_block = html.Div([
            html.P('WHO\'S FILING  ·  click a company to see its patents',
                   style={'color': MUTED, 'fontSize': '9px', 'letterSpacing': '1.5px', 'marginBottom': '7px'}),
            html.Div(filer_rows, style={'marginBottom': '14px'}),
        ])

    return html.Div([
        # Back + label
        html.Div([
            html.Button('← back', id={'type': 'back-btn', 'index': 0}, n_clicks=0, className='back-btn'),
            html.Span(f'Cluster {cid}', style={'color': MUTED, 'fontSize': '10px', 'marginLeft': 'auto'}),
            *([] if not birth else [html.Span(f'born {birth}', style={
                'color': '#0f766e', 'fontSize': '10px', 'marginLeft': '8px',
            })]),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px',
                  'padding': '10px 14px', 'borderBottom': f'1px solid {BORDER}'}),

        html.Div([
            # Growth value
            html.Div(growth_str or '', style={'marginBottom': '10px'}),
            # Conviction gauge + accelerating badge
            conv_block or '',
            # Keyword chips
            html.Div(chips, style={'marginBottom': '12px', 'lineHeight': '2'}),
            # Trend chart
            dcc.Graph(figure=build_trend(cid, horizon), config={'displayModeBar': False},
                      style={'height': '200px', 'marginBottom': '12px'}),
            # Top filers
            filer_block or '',
            # Patent titles
            *([] if not title_items else [
                html.P('SAMPLE PATENTS', style={'color': MUTED, 'fontSize': '9px',
                                                 'letterSpacing': '1.5px', 'marginBottom': '6px'}),
                html.Ul(title_items, style={'listStyle': 'none', 'padding': '0', 'margin': '0'}),
            ]),
        ], style={'padding': '14px', 'overflowY': 'auto', 'flex': '1'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})


# ── Full-screen graphical tab modals ───────────────────────────────────────────
def _modal_chrome(title, body):
    return html.Div([
        html.Div([
            html.Div([
                html.Span('◆ ', style={'color': NEON}),
                html.Span(title, style={'fontFamily': 'Space Grotesk, sans-serif',
                                          'letterSpacing': '2px', 'fontSize': '15px', 'color': TEXT}),
            ]),
            html.Button('✕  CLOSE', id={'type': 'tab-modal-close', 'index': 0}, n_clicks=0, className='info-btn'),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                  'padding': '20px 32px', 'borderBottom': f'1px solid {BORDER}'}),
        html.Div(body, style={'padding': '28px 40px', 'maxWidth': '1200px', 'margin': '0 auto'}),
    ], className='modal-overlay')


def _flow_box(title, sub, color=NEON):
    return html.Div([
        html.Div(title, style={'color': color, 'fontWeight': '700', 'fontSize': '13px',
                                'fontFamily': 'Space Grotesk', 'marginBottom': '4px'}),
        html.Div(sub, style={'color': '#94a3b8', 'fontSize': '11px', 'lineHeight': '1.4'}),
    ], style={'background': 'rgba(255,255,255,0.03)', 'border': f'1px solid {color}44',
              'borderRadius': '10px', 'padding': '14px 16px', 'flex': '1', 'minWidth': '150px',
              'boxShadow': f'0 0 18px {color}18'})


def _arrow():
    return html.Div('→', style={'color': MUTED, 'fontSize': '22px', 'alignSelf': 'center',
                                  'padding': '0 4px', 'flexShrink': '0'})


def build_tab_modal(tab):
    if tab == 'insights':
        rows = _analytics_rows()
        header = html.Div([
            html.Span('METRIC', style={'width': '240px', 'flexShrink': '0', 'color': MUTED,
                                        'fontSize': '10px', 'letterSpacing': '1px', 'fontWeight': '600'}),
            html.Span('VALUE', style={'width': '280px', 'flexShrink': '0', 'color': MUTED,
                                       'fontSize': '10px', 'letterSpacing': '1px', 'fontWeight': '600'}),
            html.Span('WHY IT MATTERS', style={'flex': '1', 'color': MUTED,
                                                'fontSize': '10px', 'letterSpacing': '1px', 'fontWeight': '600'}),
        ], style={'display': 'flex', 'gap': '16px', 'padding': '10px 0',
                  'borderBottom': f'1px solid rgba(255,255,255,0.12)'})
        body_rows = [html.Div([
            html.Span(m, style={'width': '240px', 'flexShrink': '0', 'color': TEXT,
                                 'fontSize': '14px', 'fontWeight': '600'}),
            html.Span(v, style={'width': '280px', 'flexShrink': '0', 'color': NEON,
                                 'fontSize': '15px', 'fontFamily': 'Space Grotesk, monospace', 'fontWeight': '700'}),
            html.Span(why, style={'flex': '1', 'color': '#94a3b8', 'fontSize': '12.5px', 'lineHeight': '1.5'}),
        ], style={'display': 'flex', 'gap': '16px', 'padding': '16px 0', 'alignItems': 'flex-start',
                  'borderBottom': f'1px solid rgba(255,255,255,0.05)'}) for m, v, why in rows]
        return _modal_chrome('KEY ANALYTICS — WHAT THE MODEL IS TELLING US', [
            html.P('Ranked by decision-value: the metrics that determine whether this forecast is '
                   'trustworthy and which clusters to act on.',
                   style={'color': '#94a3b8', 'fontSize': '13px', 'marginBottom': '18px'}),
            header, *body_rows,
        ])

    if tab == 'investor':
        if COMPANIES is None:
            return _modal_chrome('INVESTOR VIEW', [html.P('Company data not available.',
                                 style={'color': MUTED})])
        # ── Section 1: exposure ranking ──
        exp = COMPANIES.sort_values('exposure', ascending=False).head(25)
        exp_mx = float(exp['exposure'].max()) or 1
        def erow(i, r):
            w = r.exposure / exp_mx * 100
            return html.Div([
                html.Span(f'{i}', style={'width': '26px', 'color': '#334155', 'fontSize': '11px', 'fontFamily': 'monospace'}),
                html.Span([*([html.Span('⚡', style={'color': '#fbbf24', 'marginRight': '5px'})] if r.accelerating else []),
                           str(r.org)[:40]],
                          style={'flex': '2', 'color': TEXT, 'fontSize': '12.5px', 'fontWeight': '600'}),
                html.Span(f'{r.n_clusters}', style={'width': '70px', 'color': '#94a3b8', 'fontSize': '11px', 'textAlign': 'right'}),
                html.Span(f'{r.n_patents}', style={'width': '70px', 'color': '#94a3b8', 'fontSize': '11px', 'textAlign': 'right'}),
                html.Div(html.Div(style={'height': '4px', 'width': f'{w:.0f}%', 'background': NEON, 'borderRadius': '2px',
                         'boxShadow': f'0 0 6px {NEON}88'}), style={'width': '90px', 'marginLeft': '12px'}),
                html.Span(str(r.top_theme)[:26], style={'flex': '1.4', 'color': VIOLET, 'fontSize': '11px',
                          'marginLeft': '12px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
            ], id={'type': 'company-row-modal', 'index': str(r.org)}, className='company-row',
               n_clicks=0, title=f'Click to map {r.org} in the galaxy (closes this panel)',
               style={'display': 'flex', 'alignItems': 'center', 'padding': '5px 0',
                      'borderBottom': '1px solid rgba(255,255,255,0.04)'})
        exp_hdr = html.Div([
            html.Span('#', style={'width': '26px', 'color': MUTED, 'fontSize': '9px'}),
            html.Span('COMPANY', style={'flex': '2', 'color': MUTED, 'fontSize': '9px', 'letterSpacing': '1px'}),
            html.Span('THEMES', style={'width': '70px', 'color': MUTED, 'fontSize': '9px', 'textAlign': 'right'}),
            html.Span('PATENTS', style={'width': '70px', 'color': MUTED, 'fontSize': '9px', 'textAlign': 'right'}),
            html.Span('EXPOSURE', style={'width': '102px', 'color': MUTED, 'fontSize': '9px', 'textAlign': 'right'}),
            html.Span('TOP THEME', style={'flex': '1.4', 'color': MUTED, 'fontSize': '9px', 'marginLeft': '12px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'padding': '6px 0',
                  'borderBottom': '1px solid rgba(255,255,255,0.12)'})

        # ── Section 2: concentration (focused pure-plays) ──
        floor = COMPANIES[(COMPANIES['n_patents'] >= 10) & (COMPANIES['n_clusters'] >= 3)]
        conc = floor.sort_values('concentration', ascending=False).head(15)
        conc_rows = [html.Div([
            html.Span(str(r.org)[:40], style={'flex': '2', 'color': TEXT, 'fontSize': '12px', 'fontWeight': '600'}),
            html.Span(f'{r.concentration:.2f}', style={'width': '64px', 'color': '#10b981', 'fontSize': '12px',
                      'fontFamily': 'Space Grotesk, monospace', 'textAlign': 'right', 'fontWeight': '700'}),
            html.Span(f'{r.n_patents}p / {r.n_clusters}', style={'width': '90px', 'color': MUTED, 'fontSize': '10px', 'textAlign': 'right'}),
            html.Span(str(r.top_theme)[:30], style={'flex': '1.6', 'color': VIOLET, 'fontSize': '11px', 'marginLeft': '14px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'padding': '5px 0',
                  'borderBottom': '1px solid rgba(255,255,255,0.04)'}) for r in conc.itertuples()]

        # ── Section 3: long/short screener ──
        asg = D.get('assignees', {})
        lb2 = D['lb2']
        def filers_str(cid):
            return ', '.join(o for o, _ in (asg.get(int(cid), [])[:3])) or '—'
        def side(df_side, color, sign):
            out = []
            for r in df_side.itertuples():
                fold = np.exp(r.predicted_growth)
                lbl = str(r.auto_label).split(',')[0].strip()
                out.append(html.Div([
                    html.Div([
                        html.Span(lbl[:26], style={'color': TEXT, 'fontSize': '12px', 'fontWeight': '600'}),
                        html.Span(f'{sign}{fold:.1f}x' if sign == '' else f'{fold:.1f}x',
                                  style={'color': color, 'fontSize': '11px', 'marginLeft': 'auto',
                                         'fontFamily': 'Space Grotesk, monospace'}),
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                    html.Div(filers_str(r.cluster_id)[:48], style={'color': '#94a3b8', 'fontSize': '10px', 'marginTop': '1px'}),
                ], style={'padding': '6px 0', 'borderBottom': '1px solid rgba(255,255,255,0.04)'}))
            return out
        tail = lb2.sort_values('predicted_growth', ascending=False).head(10)
        head = lb2.sort_values('predicted_growth', ascending=True).head(10)
        ls_grid = html.Div([
            html.Div([
                html.Div('▲ TAILWINDS — accelerating themes (long ideas)',
                         style={'color': NEON, 'fontSize': '10px', 'letterSpacing': '1px', 'marginBottom': '8px', 'fontWeight': '700'}),
                *side(tail, NEON, ''),
            ], style={'flex': '1'}),
            html.Div([
                html.Div('▼ HEADWINDS — declining themes (avoid / short)',
                         style={'color': '#ef4444', 'fontSize': '10px', 'letterSpacing': '1px', 'marginBottom': '8px', 'fontWeight': '700'}),
                *side(head, '#ef4444', ''),
            ], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '28px'})

        def sec_title(t):
            return html.H3(t, style={'color': NEON, 'fontSize': '11px', 'letterSpacing': '2px',
                                     'margin': '24px 0 8px', 'textTransform': 'uppercase'})
        return _modal_chrome('INVESTOR VIEW — WHO OWNS THE EMERGING IP', [
            html.P('Company-level inversion of the patent radar. Use it to turn accelerating tech themes '
                   'into a research watchlist of companies — idea generation, not a buy signal.',
                   style={'color': '#94a3b8', 'fontSize': '13px', 'marginBottom': '6px'}),
            sec_title('1 · Company exposure ranking  (Σ patents × forecast growth)'),
            exp_hdr, *[erow(i, r) for i, r in enumerate(exp.itertuples(), 1)],
            sec_title('2 · Focused pure-plays  (concentration — corrects the “files-everywhere” bias)'),
            html.P('Patent-weighted average growth of a company’s clusters (≥10 patents, ≥3 themes). '
                   'High = IP concentrated in fast-growing areas, not spread thin.',
                   style={'color': '#64748b', 'fontSize': '11px', 'marginBottom': '8px'}),
            *conc_rows,
            sec_title('3 · Long / short theme screener'),
            ls_grid,
            html.P('Caveats: filers come from ~25 sample patents per cluster (directional, not a census); '
                   'the model forecasts patent-volume growth, not equity returns; a high theme-count means a '
                   'company files broadly, not that it’s a good bet — use concentration to find pure-plays.',
                   style={'color': '#64748b', 'fontSize': '10.5px', 'lineHeight': '1.5', 'marginTop': '18px',
                          'paddingTop': '10px', 'borderTop': '1px solid rgba(255,255,255,0.06)'}),
        ])

    if tab == 'overview':  # PIPELINE
        flow = html.Div([
            _flow_box('USPTO PATENTS', '5.6M titles, 2002–2025, quarterly', AMBER), _arrow(),
            _flow_box('EMBED', 'all-MiniLM-L6-v2 → 384-d vectors', NEON), _arrow(),
            _flow_box('CLUSTER', 'UMAP 384→10d + HDBSCAN, rolling 2-yr windows', VIOLET), _arrow(),
            _flow_box('STITCH', 'centroid cosine ≥0.85 → 2,006 global clusters', NEON), _arrow(),
            _flow_box('FEATURES', '18 time-series features / cluster / quarter', AMBER), _arrow(),
            _flow_box('FORECAST', 'GRU + LSTM → growth ranking', '#10b981'),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'marginBottom': '32px'})
        feats = ['log_count', 'count_share', 'velocity', 'acceleration', 'jerk',
                 'momentum 4Q', 'momentum 8Q', 'roll mean/std 4Q', 'roll mean/std 8Q',
                 'above_trend', 'MACD', 'z-score', 'global rank %', 'log cumsum', 'sin/cos quarter']
        chips = html.Div([html.Span(f, className='keyword-chip') for f in feats],
                         style={'lineHeight': '2.2'})
        return _modal_chrome('PIPELINE — PATENT → FORECAST', [
            flow,
            html.H3('The 18 engineered features', style={'color': NEON, 'fontSize': '11px',
                    'letterSpacing': '2px', 'marginBottom': '10px'}),
            chips,
            html.P('Patents lead publications by 1–3 years. A cluster of semantically-similar '
                   'patents accelerating today is a leading signal of the next research eruption — '
                   'detected unsupervised, with no pre-curated topic list.',
                   style={'color': '#94a3b8', 'fontSize': '13px', 'lineHeight': '1.7', 'marginTop': '24px'}),
        ])

    if tab == 'metrics':
        fig = go.Figure()
        models = ['Persistence', 'Linear trend', 'GRU', 'LSTM ★']
        for hz, off in [('2yr', NEON), ('3yr', VIOLET)]:
            vm = VAL_METRICS[hz]
            vals = [vm['Persistence (0)']['spearman'], vm['Linear trend']['spearman'],
                    vm['GRU']['spearman'], vm['LSTM']['spearman']]
            fig.add_bar(name=f'{hz} val', x=models, y=vals,
                        marker_color=off, opacity=0.92)
        # overlay test markers for LSTM
        if TEST_METRICS:
            for hz in ('2yr', '3yr'):
                t = TEST_METRICS[hz]['test']['spearman']
                fig.add_scatter(x=['LSTM ★'], y=[t], mode='markers',
                                marker=dict(symbol='diamond', size=15, color='#10b981',
                                            line=dict(color='white', width=1.5)),
                                name=f'{hz} TEST', showlegend=True)
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=TEXT, size=12), height=420,
            margin=dict(l=50, r=20, t=20, b=40),
            yaxis=dict(title='Spearman ρ (higher = better ranking)', gridcolor='rgba(255,255,255,0.06)',
                       zerolinecolor='rgba(255,255,255,0.25)'),
            legend=dict(orientation='h', y=1.08, font=dict(size=10)),
        )
        # summary cards
        def card(label, val, color):
            return html.Div([
                html.Div(val, style={'color': color, 'fontSize': '30px', 'fontWeight': '700',
                                      'fontFamily': 'Space Grotesk'}),
                html.Div(label, style={'color': MUTED, 'fontSize': '10px', 'letterSpacing': '1px',
                                        'marginTop': '4px'}),
            ], className='glass-card', style={'padding': '18px 22px', 'flex': '1', 'textAlign': 'center'})
        t2 = TEST_METRICS['2yr']['test']['spearman'] if TEST_METRICS else None
        t3 = TEST_METRICS['3yr']['test']['spearman'] if TEST_METRICS else None
        cards = html.Div([
            card('LSTM TEST ρ — 2yr', f'{t2:+.3f}' if t2 else '—', NEON),
            card('LSTM TEST ρ — 3yr', f'{t3:+.3f}' if t3 else '—', VIOLET),
            card('Linear baseline', '−0.13', '#64748b'),
            card('vs. baseline', 'beats both', '#10b981'),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '24px'})
        # ── Where the skill lives — precision at the extremes ──
        sr = SKILL_REGION
        def pcard(val, label, color, sub=''):
            return html.Div([
                html.Div(val, style={'color': color, 'fontSize': '30px', 'fontWeight': '700',
                                      'fontFamily': 'Space Grotesk'}),
                html.Div(label, style={'color': TEXT, 'fontSize': '11px', 'marginTop': '4px', 'fontWeight': '600'}),
                html.Div(sub, style={'color': MUTED, 'fontSize': '9px', 'marginTop': '2px'}),
            ], className='glass-card', style={'padding': '16px 18px', 'flex': '1', 'textAlign': 'center'})
        prec_cards = html.Div([
            pcard(f"{sr['base_rate_grow']:.0%}", 'BASE RATE', '#64748b', 'clusters that grow, unconditionally'),
            pcard(f"{sr['top10_grew']:.0%}", 'TOP-10% PREDICTED GREW', NEON, f"vs {sr['base_rate_grow']:.0%} base · ~2× lift"),
            pcard(f"{sr['bottom10_decl']:.0%}", 'BOTTOM-10% PREDICTED DECLINED', '#ef4444', 'strong directional calls'),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '14px'})

        # reliability-by-region mini bar
        rfig = go.Figure(go.Bar(
            x=['Bottom 10%', 'Middle 80%', 'Top 10%'],
            y=[sr['rho_bottom'], sr['rho_middle'], sr['rho_top']],
            marker_color=['#ef4444', '#64748b', NEON],
            text=[f"{sr['rho_bottom']:.2f}", f"{sr['rho_middle']:.2f}", f"{sr['rho_top']:.2f}"],
            textposition='outside', textfont=dict(color=TEXT)))
        rfig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color=TEXT, size=11), height=210, margin=dict(l=40, r=20, t=10, b=30),
                           yaxis=dict(title='within-band ρ', gridcolor='rgba(255,255,255,0.06)', range=[0, 0.6]),
                           showlegend=False)

        return _modal_chrome('MODEL EVALUATION — SKILL & RELIABILITY', [
            cards,
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.P('LSTM was selected per-horizon as the best model on the validation set, then '
                   'evaluated ONCE on the locked 2021–2023 test anchors. Test ρ ≈ Val ρ (slightly higher) '
                   '→ no overfitting. The linear-trend baseline is negative: naive momentum mis-ranks growth.',
                   style={'color': '#94a3b8', 'fontSize': '13px', 'lineHeight': '1.7', 'margin': '8px 0 22px'}),
            html.H3('WHERE THE SKILL LIVES', style={'color': NEON, 'fontSize': '11px', 'letterSpacing': '2px',
                    'margin': '6px 0 4px'}),
            html.P('Overall ρ ≈ 0.42 understates usefulness — what matters is directional accuracy where you '
                   'act. Only 37% of clusters grow in any window, but when the model strongly flags one it is '
                   'right ~3 times in 4. Reliability is highest at the top, solid through the middle, weakest '
                   'in fine-ordering the already-declining tail (they all sink — exact order doesn’t matter).',
                   style={'color': '#94a3b8', 'fontSize': '13px', 'lineHeight': '1.7', 'marginBottom': '14px'}),
            prec_cards,
            dcc.Graph(figure=rfig, config={'displayModeBar': False}),
            html.P('Within-band ρ is range-restricted (narrow slices have less spread, so lower ρ) — read the '
                   'precision cards as the primary evidence. Source: validation set, 2-year horizon.',
                   style={'color': '#64748b', 'fontSize': '10.5px', 'lineHeight': '1.5', 'marginTop': '6px'}),
        ])

    if tab == 'data':
        panel = D['panel']
        fig = go.Figure()
        if panel is not None:
            ts = panel.groupby('date')['count'].sum().reset_index()
            fig.add_scatter(x=ts['date'], y=ts['count'], mode='lines', fill='tozeroy',
                            line=dict(color=NEON, width=2), fillcolor='rgba(0,212,255,0.10)')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT), height=360, margin=dict(l=50, r=20, t=20, b=40),
                yaxis=dict(title='Patents / quarter', gridcolor='rgba(255,255,255,0.06)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
            )
        n_cl = len(D['labels'])
        def stat(label, val):
            return html.Div([
                html.Div(val, style={'color': NEON, 'fontSize': '26px', 'fontWeight': '700',
                                      'fontFamily': 'Space Grotesk'}),
                html.Div(label, style={'color': MUTED, 'fontSize': '10px', 'letterSpacing': '1px'}),
            ], className='glass-card', style={'padding': '16px 20px', 'flex': '1', 'textAlign': 'center'})
        stats = html.Div([
            stat('Patents', f"{int(panel['count'].sum()):,}" if panel is not None else '—'),
            stat('Clusters', f'{n_cl:,}'),
            stat('Quarters', str(len(panel['date'].unique())) if panel is not None else '—'),
            stat('Years', '2002–2025'),
        ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '24px'})
        return _modal_chrome('DATA SOURCE — USPTO PATENT CORPUS', [
            stats,
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            html.Div([
                html.Span('Train ', style={'color': NEON}), '2002–2017   ',
                html.Span('Val ', style={'color': AMBER}), '2018–2020   ',
                html.Span('Test ', style={'color': VIOLET}), '2021–2023 (locked)   ',
                html.Span('Forecast ', style={'color': '#10b981'}), '2024 →',
            ], style={'color': '#94a3b8', 'fontSize': '13px', 'marginTop': '16px',
                      'fontFamily': 'Space Grotesk', 'letterSpacing': '1px'}),
        ])

    # retrospective
    retro_path = CDIR / 'retrospective_matches.csv'
    if not retro_path.exists():
        return _modal_chrome('RETROSPECTIVE VALIDATION', [
            html.P('Retrospective validation not yet computed.', style={'color': MUTED, 'fontSize': '14px'}),
            html.P('Run scripts/match_eruptions.py to match the 62 known eruption events to '
                   'patent clusters and measure lead time.', style={'color': '#64748b', 'fontSize': '12px'})])

    df = pd.read_csv(retro_path)
    n_total   = len(df)
    strong    = int((df['cosine_sim'] >= 0.5).sum())
    led       = df[df['lead_time_q'] > 0]
    n_led     = len(led)
    med_lead  = led['lead_time_q'].median() / 4 if n_led else 0

    def card(val, label, color):
        return html.Div([
            html.Div(val, style={'color': color, 'fontSize': '28px', 'fontWeight': '700',
                                  'fontFamily': 'Space Grotesk'}),
            html.Div(label, style={'color': MUTED, 'fontSize': '10px', 'letterSpacing': '1px',
                                    'marginTop': '4px', 'lineHeight': '1.3'}),
        ], className='glass-card', style={'padding': '16px 20px', 'flex': '1', 'textAlign': 'center'})

    cards = html.Div([
        card(str(n_total), 'KNOWN ERUPTIONS<br>MATCHED', NEON),
        card(f'{strong}/{n_total}', 'STRONG SEMANTIC<br>MATCH (cos≥0.5)', VIOLET),
        card(f'{n_led}', f'PATENTS LED<br>PUBLICATIONS ({100*n_led//n_total}%)', '#10b981'),
        card(f'{med_lead:.0f}yr', 'MEDIAN LEAD<br>(when leading)', AMBER),
    ], style={'display': 'flex', 'gap': '14px', 'marginBottom': '22px'})

    # scatter: cosine match vs lead time
    fig = go.Figure()
    sc = df.copy()
    sc['lead_yr'] = sc['lead_time_q'] / 4
    fig.add_scatter(x=sc['cosine_sim'], y=sc['lead_yr'], mode='markers',
                    marker=dict(size=9, color=sc['lead_yr'], colorscale=GROWTH_CS,
                                line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
                    text=sc['topic_name'],
                    hovertemplate='<b>%{text}</b><br>cosine %{x:.2f}<br>lead %{y:.1f}yr<extra></extra>')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color=TEXT), height=240, margin=dict(l=50, r=20, t=10, b=40),
                      xaxis=dict(title='Semantic match (cosine)', gridcolor='rgba(255,255,255,0.06)'),
                      yaxis=dict(title='Lead time (years)', gridcolor='rgba(255,255,255,0.06)',
                                 zerolinecolor='rgba(255,255,255,0.2)'))

    # top-15 confident matches table
    top = df.sort_values('cosine_sim', ascending=False).head(15)
    hdr = html.Div([
        html.Span('RESEARCH ERUPTION', style={'flex': '2', 'color': MUTED, 'fontSize': '9px', 'letterSpacing': '1px'}),
        html.Span('MATCHED PATENT CLUSTER', style={'flex': '2', 'color': MUTED, 'fontSize': '9px', 'letterSpacing': '1px'}),
        html.Span('COSINE', style={'width': '60px', 'color': MUTED, 'fontSize': '9px', 'textAlign': 'right'}),
        html.Span('ERUPT', style={'width': '54px', 'color': MUTED, 'fontSize': '9px', 'textAlign': 'right'}),
        html.Span('LEAD', style={'width': '60px', 'color': MUTED, 'fontSize': '9px', 'textAlign': 'right'}),
    ], style={'display': 'flex', 'gap': '10px', 'padding': '6px 0',
              'borderBottom': f'1px solid rgba(255,255,255,0.12)'})
    rows = []
    for _, r in top.iterrows():
        lead = r['lead_time_q']
        lead_str = f'{lead/4:+.1f}yr' if pd.notna(lead) and lead > 0 else '—'
        lead_col = '#10b981' if pd.notna(lead) and lead > 0 else MUTED
        rows.append(html.Div([
            html.Span(str(r['topic_name'])[:38], style={'flex': '2', 'color': TEXT, 'fontSize': '11px'}),
            html.Span(str(r['auto_label']).split(',')[0][:28], style={'flex': '2', 'color': '#94a3b8', 'fontSize': '11px'}),
            html.Span(f"{r['cosine_sim']:.2f}", style={'width': '60px', 'color': NEON, 'fontSize': '11px',
                       'textAlign': 'right', 'fontFamily': 'monospace'}),
            html.Span(str(int(r['eruption_year'])), style={'width': '54px', 'color': '#94a3b8', 'fontSize': '11px',
                       'textAlign': 'right', 'fontFamily': 'monospace'}),
            html.Span(lead_str, style={'width': '60px', 'color': lead_col, 'fontSize': '11px',
                       'textAlign': 'right', 'fontFamily': 'monospace'}),
        ], style={'display': 'flex', 'gap': '10px', 'padding': '5px 0',
                  'borderBottom': f'1px solid rgba(255,255,255,0.04)'}))

    return _modal_chrome('RETROSPECTIVE VALIDATION — DO PATENTS LEAD RESEARCH?', [
        html.P('Independent test: we matched 62 known research eruptions (from OpenAlex, e.g. CRISPR, '
               'perovskites, GANs) to their nearest patent cluster, then measured whether the patents '
               'took off first. This validates the core premise — patents as a leading indicator.',
               style={'color': '#94a3b8', 'fontSize': '13px', 'lineHeight': '1.6', 'marginBottom': '18px'}),
        cards,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.P('Top 15 matches by semantic similarity', style={'color': MUTED, 'fontSize': '10px',
                'letterSpacing': '1.5px', 'margin': '18px 0 4px'}),
        hdr, *rows,
        html.P('Note: median lead is inflated by an eager takeoff threshold on long-lived clusters; the '
               'robust signals are the strong semantic match and that ~60% of eruptions had patents rising first '
               '(Graphene led ~1yr, Quantum Dots ~2yr).',
               style={'color': '#64748b', 'fontSize': '10.5px', 'lineHeight': '1.5', 'marginTop': '14px',
                      'paddingTop': '10px', 'borderTop': f'1px solid rgba(255,255,255,0.06)'}),
    ])


# ── Methodology modal ──────────────────────────────────────────────────────────
def _disc(title, body):
    return html.Div([
        html.Div(title, style={'color': TEXT, 'fontSize': '12px', 'fontWeight': '700', 'marginBottom': '3px'}),
        html.Div(body, style={'color': '#94a3b8', 'fontSize': '11px', 'lineHeight': '1.55'}),
    ])


def build_modal():
    def section(title, content):
        return html.Div([html.H3(title), content], className='modal-section')

    def row(label, val, cls=''):
        return html.Div([
            html.Span(label, className='metric-label'),
            html.Span(val, className=f'metric-value {cls}'),
        ], className='metric-row')

    lb2 = D['lb2']
    n_clusters = len(D['labels']) or len(D['coords3d'])
    n_quarters = len(D['panel']['date'].unique()) if D['panel'] is not None else '—'
    n_patents  = int(D['panel']['count'].sum()) if D['panel'] is not None else '—'

    pipeline_steps = [
        ('1', html.Span([html.Strong('Embed'), ' USPTO patent titles → 384-d vectors (all-MiniLM-L6-v2)'])),
        ('2', html.Span([html.Strong('Cluster'), ' UMAP(384→10) + HDBSCAN on rolling 2-yr windows'])),
        ('3', html.Span([html.Strong('Stitch'), ' cross-window centroid matching (cosine ≥ 0.85) → persistent global cluster IDs'])),
        ('4', html.Span([html.Strong('Features'), ' 18 engineered per cluster per quarter (velocity, momentum, MACD, z-score, …)'])),
        ('5', html.Span([html.Strong('Split'), ' train anchors → 2005–2017 | val → 2018–2020 | test locked 2021–2023'])),
        ('6', html.Span([html.Strong('Model'), ' GRU + LSTM, target = log_count[t+H] − log_count[t], Huber loss, Spearman metric'])),
        ('7', html.Span([html.Strong('Forecast'), ' best-of-two model scores all clusters → ranked leaderboard'])),
    ]
    steps_html = [html.Div([
        html.Div(num, className='step-num'),
        html.Div(desc, className='step-body'),
    ], className='pipeline-step') for num, desc in pipeline_steps]

    return html.Div([
        # Header bar
        html.Div([
            html.Div([
                html.Span('◆', style={'color': NEON, 'marginRight': '10px'}),
                html.Span('METHODOLOGY & MODEL CARD', style={
                    'fontFamily': 'Space Grotesk, sans-serif',
                    'fontWeight': '700', 'letterSpacing': '2px', 'fontSize': '13px',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Button('✕  CLOSE', id={'type': 'modal-close', 'index': 0}, n_clicks=0, className='info-btn'),
        ], style={
            'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
            'padding': '16px 28px', 'borderBottom': f'1px solid {BORDER}',
            'position': 'sticky', 'top': '0', 'background': 'rgba(7,11,20,0.97)', 'zIndex': '10',
        }),

        # Grid content
        html.Div([
            section('7-STEP PIPELINE', html.Div(steps_html)),

            section('DATA SOURCE', html.Div([
                row('Source', 'USPTO PatentsView g_patent.tsv'),
                row('Years covered', '2002 – 2025'),
                row('Total patents (assigned to clusters)', f'{n_patents:,}' if isinstance(n_patents, int) else n_patents),
                row('Active clusters (≥16 quarters)', str(n_clusters)),
                row('Quarters in panel', str(n_quarters)),
                row('Embedding model', 'all-MiniLM-L6-v2 (384-d)'),
                row('Cluster algorithm', 'UMAP(384→10) + HDBSCAN'),
            ])),

            section('MODEL ARCHITECTURE', html.Div([
                row('Input shape', 'Input(8 quarters, 18 features)'),
                row('Layer 1', 'RNN(64 units, return_sequences=True)'),
                row('Dropout', '0.2'),
                row('Layer 2', 'RNN(32 units)'),
                row('Dropout', '0.2'),
                row('Output', 'Dense(1) → growth delta'),
                row('Architectures trained', 'GRU and LSTM (best selected per horizon)'),
                row('Horizons', '2yr (8Q) model · 3yr (12Q) model · 5yr linear extrapolation'),
            ])),

            section('TRAINING CONFIG', html.Div([
                row('Target variable', 'log_count[t+H] − log_count[t]', 'good'),
                row('Loss function', 'Huber (δ=1.0) — robust to outlier clusters'),
                row('Primary metric', 'Spearman rank correlation', 'good'),
                row('Secondary metrics', 'RMSE, MAE'),
                row('Optimiser', 'Adam, LR=3e-4'),
                row('Epochs / batch', '80 max · batch 64'),
                row('Early stopping', 'patience=10, restore best weights'),
                row('LR schedule', 'ReduceLROnPlateau factor=0.5, patience=5'),
            ])),

            section('BASELINES', html.Div([
                html.P('Model must beat both baselines to add value:', style={'color': MUTED, 'fontSize': '12px', 'marginBottom': '10px'}),
                row('Persistence', 'Predict 0 growth for every cluster'),
                row('Linear trend', 'Fit slope to last 8Q, extrapolate × H'),
                html.P('If GRU/LSTM Spearman > Linear trend Spearman → model learned something beyond extrapolation.',
                       style={'color': '#334155', 'fontSize': '11px', 'marginTop': '10px', 'lineHeight': '1.5'}),
            ])),

            section('TRAIN / VAL / TEST SPLIT', html.Div([
                row('Train anchors', 'Q4 2006 → Q4 2017', 'good'),
                row('Val anchors', 'Q1 2018 → Q4 2020', 'good'),
                row('Test anchors (2yr)', 'Q1 2021 → Q4 2023  —  LOCKED', 'warn'),
                row('Test anchors (3yr)', 'Q1 2021 → Q4 2022  —  LOCKED', 'warn'),
                row('Leaderboard anchors', 'Most recent 8Q → predicts 2026–2027'),
                html.P('Test set is never evaluated during model development. Leaderboard predictions are forward-only.',
                       style={'color': '#334155', 'fontSize': '11px', 'marginTop': '10px', 'lineHeight': '1.5'}),
            ])),

            section('5-YEAR FORECAST NOTE', html.Div([
                html.P('The 5yr column is a linear extrapolation of recent patent velocity — not a model prediction. '
                       'It is shown with reduced opacity and labelled speculative.',
                       style={'color': MUTED, 'fontSize': '12px', 'lineHeight': '1.6'}),
                html.P('A 5yr model would require targets in 2028–2030, which lie beyond available data. '
                       'Linear extrapolation is presented as a directional signal only.',
                       style={'color': '#334155', 'fontSize': '11px', 'marginTop': '8px', 'lineHeight': '1.6'}),
            ])),

        ], className='modal-grid'),

        # ── Disclaimer & limitations (full-width) ──
        html.Div([
            html.H3('⚠  DISCLAIMER & LIMITATIONS', style={
                'color': AMBER, 'fontSize': '12px', 'letterSpacing': '2px',
                'textTransform': 'uppercase', 'margin': '0 0 14px 0'}),
            html.Div([
                _disc('Not investment advice',
                      'This is a research and idea-generation tool. Nothing here is a recommendation to buy, '
                      'sell, or hold any security. Do your own diligence; consult a licensed professional.'),
                _disc('Forecasts tech volume, not returns',
                      'The model predicts growth in patent filing volume — a leading R&D signal. It does NOT '
                      'predict revenue, earnings, or stock price. Patents ≠ products ≠ profit.'),
                _disc('Modest, honest skill',
                      'Test Spearman ≈ 0.42 / 0.47 means the ranking is meaningfully better than chance, not '
                      'precise. Treat it as direction-of-travel across many clusters, not a per-cluster guarantee. '
                      'Skill is highest at the extremes and near-zero in the middle of the ranking.'),
                _disc('Company filers are sampled',
                      '"Who\'s filing" and the investor view are built from ~25 representative sample patents per '
                      'cluster (the full per-patent map was not persisted). Company names are reliable; counts and '
                      'rankings are directional, not an exhaustive census. Broad filers (e.g. IBM) appear in many '
                      'clusters by construction — use the concentration score to find true pure-plays.'),
                _disc('Conviction is a heuristic',
                      'The 0–100 conviction score is a hand-weighted blend (50% growth, 25% volume, 25% '
                      'acceleration), not a calibrated probability. It is a triage aid for where to look first.'),
                _disc('Auto-labels are keywords',
                      'Cluster names are TF-IDF keywords from patent titles, not curated taxonomy. Some are niche '
                      'or mechanical; read the sample patents to judge what a cluster really is.'),
                _disc('Galaxy positions are approximate',
                      'Node positions come from a UMAP projection of embeddings — local neighbourhoods (nearby = '
                      'related) are meaningful; absolute distances and global layout are not.'),
                _disc('Snapshot in time',
                      'Built on USPTO filings 2002–2025. Patent data lags filing by ~18 months; recent quarters are '
                      'undercounted. Re-run the pipeline on fresh data to refresh.'),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '14px'}),
        ], className='modal-section', style={'margin': '16px 24px 28px', 'borderColor': 'rgba(245,158,11,0.25)'}),
    ], className='modal-overlay', id='modal-overlay')


# ── First-visit guided tour ─────────────────────────────────────────────────────
TOUR_STEPS = [
    ('① The Technology Galaxy',
     'Every star is a patent cluster — a tech theme discovered automatically from 5.6M USPTO patents. '
     'Colour = forecast growth (red ↓ declining · cyan ↑ growing), size = patent volume, position = '
     'similarity (nearby stars are related tech). Drag to rotate, scroll to zoom, click a star to inspect.'),
    ('② Emerging Clusters — the ranking',
     'The right panel ranks clusters by forecast growth for the chosen horizon (2YR / 3YR up top). '
     'Click any row — or any star — to see its trend, conviction score, the companies filing in it, and '
     'real patents you can open. Use ⬇ WATCHLIST to export the top picks.'),
    ('③ Intelligence tabs (bottom)',
     'INSIGHTS = headline results · INVESTOR = who owns the emerging IP (click a company to light up its '
     'clusters) · METRICS = model skill vs baselines · RETROSPECTIVE = does it actually lead real research? '
     'Hit ⛶ to expand any tab full-screen. Top-right: METHODOLOGY & DISCLAIMER — read it first.'),
]

def build_tour(step):
    """step is 1-based; returns the overlay card or [] when finished."""
    if not (1 <= step <= len(TOUR_STEPS)):
        return []
    title, body = TOUR_STEPS[step - 1]
    last = step == len(TOUR_STEPS)
    dots = html.Div([
        html.Span('●' if i == step - 1 else '○',
                  style={'color': NEON if i == step - 1 else MUTED, 'fontSize': '10px', 'marginRight': '4px'})
        for i in range(len(TOUR_STEPS))
    ], style={'marginBottom': '10px'})
    return html.Div(html.Div([
        dots,
        html.Div(title, style={'color': TEXT, 'fontFamily': 'Space Grotesk, sans-serif',
                               'fontSize': '17px', 'fontWeight': '700', 'marginBottom': '8px'}),
        html.Div(body, style={'color': '#94a3b8', 'fontSize': '13px', 'lineHeight': '1.65', 'marginBottom': '18px'}),
        html.Div([
            html.Button('Skip tour', id={'type': 'tour-btn', 'step': step, 'action': 'skip'},
                        n_clicks=0, className='tour-skip'),
            html.Button('Got it ✓' if last else 'Next →',
                        id={'type': 'tour-btn', 'step': step, 'action': 'done' if last else 'next'},
                        n_clicks=0, className='tour-next'),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
    ], className='tour-card'), className='tour-backdrop')


# ── Time slider marks ──────────────────────────────────────────────────────────
def _slider_config():
    if D['panel'] is None:
        return 2002, 2025, {2002: '2002', 2010: '2010', 2018: '2018', 2025: '2025'}, 2025
    years = sorted(int(y) for y in D['panel']['date'].dt.year.unique())
    marks = {y: str(y) for y in years if y % 4 == 0}
    return years[0], years[-1], marks, years[-1]


# ── Layout ─────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='Patent Emerging Radar', update_title=None,
                suppress_callback_exceptions=True)
app.config.suppress_callback_exceptions = True

s_min, s_max, s_marks, s_def = _slider_config()

app.layout = html.Div([

    # ── Header ─────────────────────────────────────────────────────────────
    html.Div([
        html.Span('◆', className='logo-mark'),
        html.Span('PATENT EMERGING RADAR', className='app-title'),
        html.Span('Technology Foresight · research tool, not investment advice', className='app-subtitle'),

        html.Div([
            dcc.RadioItems(id='horizon-sel', inline=True, value='2yr', className='horizon-radio',
                           options=[{'label': '2 YR', 'value': '2yr'},
                                    {'label': '3 YR', 'value': '3yr'},
                                    {'label': '5 YR', 'value': '5yr'}]),
        ], style={'marginLeft': 'auto', 'marginRight': '16px'}),

        html.Div([
            dcc.RadioItems(id='color-mode', inline=True, value='growth', className='horizon-radio',
                           options=[{'label': 'GROWTH', 'value': 'growth'},
                                    {'label': 'AGE', 'value': 'birth'},
                                    {'label': 'VELOCITY', 'value': 'velocity'}]),
        ], style={'marginRight': '16px'}),

        html.Button('METHODOLOGY & DISCLAIMER ↗', id='modal-open-btn', n_clicks=0, className='info-btn'),
    ], className='app-header'),

    # ── Main area ───────────────────────────────────────────────────────────
    html.Div([

        # Galaxy + time scrubber
        html.Div([
            dcc.Graph(id='galaxy-3d', figure=build_galaxy(),
                      config={'displayModeBar': False},
                      style={'height': 'calc(100% - 64px)', 'width': '100%'}),

            # ── Collapsible (i) encoding key — click to expand ──
            html.Details([
                html.Summary([
                    html.Span('ⓘ', style={'fontSize': '13px', 'marginRight': '6px'}),
                    html.Span('How to read the galaxy'),
                ], className='info-summary'),
                html.Div([
                    html.Div([html.Span('◍', style={'color': TEXT, 'marginRight': '8px', 'fontSize': '13px'}),
                              html.Span('Position', style={'color': TEXT, 'fontWeight': '600'}),
                              html.Span(' — semantic similarity (UMAP of patent embeddings). Near = related tech.',
                                        style={'color': '#94a3b8'})],
                             style={'fontSize': '10.5px', 'marginBottom': '5px', 'lineHeight': '1.4'}),
                    html.Div([html.Span('⬤', style={'color': TEXT, 'marginRight': '8px', 'fontSize': '11px'}),
                              html.Span('Size', style={'color': TEXT, 'fontWeight': '600'}),
                              html.Span(' — patent volume (recent filings in that cluster).',
                                        style={'color': '#94a3b8'})],
                             style={'fontSize': '10.5px', 'marginBottom': '5px', 'lineHeight': '1.4'}),
                    html.Div([html.Span('●', style={'color': '#ef4444', 'marginRight': '2px', 'fontSize': '11px'}),
                              html.Span('●', style={'color': '#f59e0b', 'marginRight': '6px', 'fontSize': '11px'}),
                              html.Span('●', style={'color': '#22d3ee', 'marginRight': '2px', 'fontSize': '11px'}),
                              html.Span('●', style={'color': '#5eead4', 'marginRight': '8px', 'fontSize': '11px'}),
                              html.Span('Colour', style={'color': TEXT, 'fontWeight': '600'}),
                              html.Span(' — forecast growth: ', style={'color': '#94a3b8'}),
                              html.Span('red ↓ decline', style={'color': '#ef4444'}),
                              html.Span('  /  ', style={'color': MUTED}),
                              html.Span('cyan ↑ growth', style={'color': '#22d3ee'})],
                             style={'fontSize': '10.5px', 'lineHeight': '1.4'}),
                    html.Div('Drag to rotate · scroll to zoom · click a star to inspect',
                             style={'color': MUTED, 'fontSize': '9px', 'marginTop': '8px',
                                    'fontStyle': 'italic', 'borderTop': f'1px solid {BORDER}', 'paddingTop': '7px'}),
                ], style={'marginTop': '10px', 'maxWidth': '320px'}),
            ], className='info-details', style={
                'position': 'absolute', 'top': '14px', 'left': '14px', 'zIndex': '20'}),
            html.Div([
                html.Div([
                    html.Span('TIME', style={'color': MUTED, 'fontSize': '9px', 'letterSpacing': '1.5px',
                                              'marginRight': '12px', 'flexShrink': '0'}),
                    html.Div(
                        dcc.Slider(id='time-slider', min=s_min, max=s_max, step=1,
                                   value=s_def, marks=s_marks,
                                   tooltip={'placement': 'top', 'always_visible': False},
                                   className='time-slider'),
                        style={'flex': '1', 'minWidth': '0'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'flex': '1', 'width': '100%',
                          'padding': '0 24px 0 16px'}),
            ], style={'height': '64px', 'display': 'flex', 'alignItems': 'center', 'width': '100%',
                      'flexShrink': '0', 'overflow': 'visible', 'borderTop': f'1px solid {BORDER}'}),
        ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column',
                  'borderRight': f'1px solid {BORDER}', 'overflow': 'hidden', 'position': 'relative',
                  'background': 'radial-gradient(ellipse at 50% 45%, #0d1426 0%, #0a0f1d 60%, #070b14 100%)'}),

        # Right panel
        html.Div([
            # Search bar
            html.Div([
                dcc.Input(id='search-input', type='text', debounce=True,
                          placeholder='Search keywords, tech areas, or a US patent #…',
                          style={
                              'width': '100%', 'background': 'rgba(255,255,255,0.04)',
                              'border': f'1px solid {BORDER}', 'borderRadius': '6px',
                              'padding': '7px 12px', 'color': TEXT, 'fontSize': '12px',
                              'outline': 'none', 'fontFamily': 'inherit',
                          }),
            ], style={'padding': '10px 12px', 'borderBottom': f'1px solid {BORDER}'}),

            # Domain chips
            html.Div([
                html.Button('ALL', id='domain-all', n_clicks=0,
                            className='domain-chip domain-chip--active'),
                *[html.Button(d, id={'type': 'domain-chip', 'index': d},
                              n_clicks=0, className='domain-chip')
                  for d in ALL_DOMAINS],
            ], id='domain-chips-row', style={
                'display': 'flex', 'gap': '5px', 'flexWrap': 'nowrap',
                'overflowX': 'auto', 'padding': '8px 12px',
                'borderBottom': f'1px solid {BORDER}',
            }),

            html.Div([
                html.Span('EMERGING CLUSTERS', id='right-panel-header-text', n_clicks=0,
                          title='Click to return to the full ranking', style={'cursor': 'pointer'}),
                html.Button('⬇ WATCHLIST', id='export-btn', n_clicks=0, className='export-btn'),
            ], id='right-panel-header', className='panel-header',
               style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}),
            html.Div(
                dcc.RadioItems(id='lb-direction', value='growers', className='dir-toggle',
                               options=[{'label': '▲ TOP GROWERS', 'value': 'growers'},
                                        {'label': '▼ TOP DECLINERS', 'value': 'decliners'}]),
                style={'padding': '6px 12px', 'borderBottom': f'1px solid {BORDER}'}),
            dcc.Download(id='watchlist-download'),
            html.Div(id='right-panel-content',
                     children=build_leaderboard('2yr'),
                     style={'flex': '1', 'overflow': 'hidden'}),
        ], style={'width': '370px', 'flexShrink': '0', 'display': 'flex',
                  'flexDirection': 'column', 'overflow': 'hidden'}),

    ], id='main-area', style={
        'display': 'flex', 'flexDirection': 'row',
        'position': 'fixed', 'top': '52px', 'left': '0', 'right': '0', 'bottom': '185px',
    }),

    # ── Intelligence strip ──────────────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Tabs(id='intel-tabs', value='insights', className='intel-tabs',
                     style={'borderBottom': f'1px solid {BORDER}', 'flex': '1'},
                     children=[
                         dcc.Tab(label='INSIGHTS',      value='insights',      className='tab', selected_className='tab--selected'),
                         dcc.Tab(label='INVESTOR',      value='investor',      className='tab', selected_className='tab--selected'),
                         dcc.Tab(label='PIPELINE',      value='overview',      className='tab', selected_className='tab--selected'),
                         dcc.Tab(label='METRICS',       value='metrics',       className='tab', selected_className='tab--selected'),
                         dcc.Tab(label='DATA',          value='data',          className='tab', selected_className='tab--selected'),
                         dcc.Tab(label='RETROSPECTIVE', value='retrospective', className='tab', selected_className='tab--selected'),
                     ]),
            html.Button('⛶  EXPAND', id='tab-expand-btn', n_clicks=0, className='info-btn',
                        style={'margin': '0 8px 0 16px', 'flexShrink': '0'}),
            html.Button('▾', id='intel-collapse-btn', n_clicks=0, className='info-btn',
                        title='Collapse / expand this panel',
                        style={'margin': '0 12px 0 0', 'flexShrink': '0', 'fontSize': '13px',
                               'padding': '5px 12px'}),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'borderBottom': f'1px solid {BORDER}'}),
        html.Div(id='intel-content', style={'padding': '10px 20px', 'overflowY': 'auto', 'height': '130px'}),
    ], id='intel-strip', style={
        'position': 'fixed', 'bottom': '0', 'left': '0', 'right': '0', 'height': '185px',
        'borderTop': f'1px solid {BORDER}',
        'background': 'rgba(7,11,20,0.96)',
    }),

    # ── Modal (hidden by default) ───────────────────────────────────────────
    html.Div(id='modal-container', children=[]),
    html.Div(id='tab-modal-container', children=[]),

    # ── Guided tour (first visit only) ──────────────────────────────────────
    html.Div(id='tour-container', children=[]),
    dcc.Interval(id='tour-init', interval=800, max_intervals=1),

    # ── Stores ─────────────────────────────────────────────────────────────
    dcc.Store(id='selected-cluster', data=None),
    dcc.Store(id='modal-open', data=False),
    dcc.Store(id='active-domain', data=None),
    dcc.Store(id='filter-ids', data=None),
    dcc.Store(id='tour-seen', data=False, storage_type='local'),

], style={'background': BG, 'minHeight': '100vh',
          'fontFamily': 'Inter, -apple-system, sans-serif', 'color': TEXT})


# ── Callbacks ──────────────────────────────────────────────────────────────────

# ── Filter: domain chip selection ──────────────────────────────────────────────
@app.callback(
    Output('active-domain', 'data'),
    Output('domain-chips-row', 'children'),
    Input('domain-all', 'n_clicks'),
    Input({'type': 'domain-chip', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def select_domain(all_clicks, chip_clicks):
    tid = ctx.triggered_id
    if tid == 'domain-all' or tid is None:
        active = None
    else:
        active = tid['index']

    def chip_cls(d):
        return 'domain-chip domain-chip--active' if d == active else 'domain-chip'

    chips = [
        html.Button('ALL', id='domain-all', n_clicks=0,
                    className='domain-chip' + (' domain-chip--active' if active is None else '')),
        *[html.Button(d, id={'type': 'domain-chip', 'index': d}, n_clicks=0,
                      className=chip_cls(d)) for d in ALL_DOMAINS],
    ]
    return active, chips


# ── Filter: compute matching cluster IDs ───────────────────────────────────────
@app.callback(
    Output('filter-ids', 'data'),
    Input('search-input', 'value'),
    Input('active-domain', 'data'),
)
def compute_filter(search, domain):
    ids = _filter_ids(search, domain)
    return list(ids) if ids is not None else None


# ── Click a company (INVESTOR tab) → light up its clusters in the galaxy ─────────
@app.callback(
    Output('filter-ids', 'data', allow_duplicate=True),
    Output('selected-cluster', 'data', allow_duplicate=True),
    Output('tab-modal-container', 'children', allow_duplicate=True),
    Input({'type': 'company-row', 'index': ALL}, 'n_clicks'),
    Input({'type': 'company-row-modal', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def filter_by_company(strip_clicks, modal_clicks):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    # ignore the zero-click events fired when rows are (re)rendered
    if not any((strip_clicks or []) + (modal_clicks or [])):
        raise dash.exceptions.PreventUpdate
    org  = trig['index']
    cids = COMPANY_CLUSTERS.get(org, [])
    if not cids:
        raise dash.exceptions.PreventUpdate
    # set galaxy filter, clear any selected cluster, close the expand modal
    return [int(c) for c in cids], None, []


# ── Guided tour (shows once; dismissal persisted in localStorage) ────────────────
@app.callback(
    Output('tour-container', 'children'),
    Output('tour-seen', 'data'),
    Input('tour-init', 'n_intervals'),
    Input({'type': 'tour-btn', 'step': ALL, 'action': ALL}, 'n_clicks'),
    State('tour-seen', 'data'),
    prevent_initial_call=True,
)
def run_tour(n_init, btn_clicks, seen):
    trig = ctx.triggered_id
    if trig == 'tour-init':
        if seen:
            return [], dash.no_update
        return build_tour(1), dash.no_update
    if isinstance(trig, dict) and trig.get('type') == 'tour-btn':
        if not any(btn_clicks or []):
            raise dash.exceptions.PreventUpdate
        if trig['action'] in ('skip', 'done'):
            return [], True
        nxt = trig['step'] + 1
        if nxt > len(TOUR_STEPS):
            return [], True
        return build_tour(nxt), dash.no_update
    raise dash.exceptions.PreventUpdate


# ── Galaxy: update on any input ────────────────────────────────────────────────
@app.callback(
    Output('galaxy-3d', 'figure'),
    Input('horizon-sel', 'value'),
    Input('color-mode', 'value'),
    Input('time-slider', 'value'),
    Input('filter-ids', 'data'),
    prevent_initial_call=False,
)
def update_galaxy(horizon, color_mode, time_year, filter_ids_list):
    fids = set(filter_ids_list) if filter_ids_list is not None else None
    if ctx.triggered_id in ('horizon-sel', 'color-mode', 'time-slider', 'filter-ids'):
        sizes, colors, cs = _node_props(horizon, color_mode, time_year, fids)
        if not sizes:
            return build_galaxy(horizon, color_mode, time_year, fids)
        p = Patch()
        p['data'][0]['marker']['color']      = colors
        p['data'][0]['marker']['size']       = sizes
        p['data'][0]['marker']['colorscale'] = cs
        return p
    return build_galaxy(horizon, color_mode, time_year, fids)


def build_area_chart(filter_ids, horizon):
    """Horizontal bar chart: top growers (cyan) + top decliners (red) in a tech area."""
    lb = _lb(horizon)
    if lb is None or not filter_ids:
        return None
    gcol = _gcol(horizon)
    sub = lb[lb['cluster_id'].astype(int).isin(filter_ids)].copy()
    if sub.empty:
        return None
    sub['g'] = sub[gcol]
    grow = sub.nlargest(5, 'g')
    decl = sub.nsmallest(5, 'g')
    decl = decl[decl['g'] < 0]                      # only real decliners
    rows = pd.concat([decl.iloc[::-1], grow.iloc[::-1]]).drop_duplicates('cluster_id')
    if rows.empty:
        return None
    names = [str(l).split(',')[0][:22] for l in rows['auto_label']]
    vals  = rows['g'].tolist()
    colors = [NEON if v > 0 else '#ef4444' for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker=dict(color=colors), hovertemplate='%{y}: %{x:+.2f}<extra></extra>'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT, size=9), height=max(150, 22*len(rows)+40),
        margin=dict(l=4, r=10, t=6, b=20),
        xaxis=dict(title='', gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.25)',
                   tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=9, color='#cbd5e1'), automargin=True),
        bargap=0.35, showlegend=False)
    return dcc.Graph(figure=fig, config={'displayModeBar': False},
                     style={'marginBottom': '6px'})


def build_filtered_list(filter_ids, horizon, direction='growers'):
    """Right panel content when a search/domain filter is active."""
    lb = _lb(horizon)
    gcol = _gcol(horizon)
    labels = D['labels']
    domains = DOMAINS

    if not filter_ids:
        return html.Div(
            html.P('No clusters match.', style={'color': MUTED, 'textAlign': 'center',
                                                 'marginTop': '40px', 'fontSize': '12px'})
        )

    rows = []
    # Sort by leaderboard growth if available, else alphabetically
    if lb is not None:
        lb_map = dict(zip(lb['cluster_id'].astype(int), lb[gcol]))
        sorted_ids = sorted(filter_ids, key=lambda c: lb_map.get(c, 0),
                            reverse=(direction != 'decliners'))
    else:
        sorted_ids = sorted(filter_ids,
                            key=lambda c: labels.get(c, f'Cluster {c}'))

    for cid in sorted_ids[:40]:
        lbl = labels.get(cid, f'Cluster {cid}')
        dom = domains.get(cid, 'Other')
        g   = lb_map.get(cid, None) if lb is not None else None
        g_str = f'{g:+.3f}' if g is not None else '—'
        g_col = NEON if (g or 0) > 0 else VIOLET

        rows.append(html.Div([
            html.Div([
                html.Span(lbl[:30] + ('…' if len(lbl) > 30 else ''),
                          style={'fontSize': '12px', 'color': TEXT,
                                 'overflow': 'hidden', 'textOverflow': 'ellipsis',
                                 'whiteSpace': 'nowrap'}),
                html.Span(dom, style={'fontSize': '10px', 'color': MUTED,
                                       'marginLeft': '6px'}),
            ], style={'flex': '1', 'overflow': 'hidden'}),
            html.Span(g_str, style={'color': g_col, 'fontSize': '11px',
                                     'fontFamily': 'Space Grotesk, monospace',
                                     'flexShrink': '0'}),
        ], id={'type': 'lb-row', 'index': cid}, className='lb-row', n_clicks=0))

    chart = build_area_chart(filter_ids, horizon)
    children = []
    if chart is not None:
        children.append(html.Div([
            html.Div('FORECAST — GROWERS ▲ / DECLINERS ▼ IN THIS AREA',
                     style={'color': '#475569', 'fontSize': '9px', 'letterSpacing': '1px',
                            'padding': '8px 12px 2px'}),
            chart,
        ]))
    children.append(html.Div(rows))
    return html.Div(children, style={'overflowY': 'auto', 'height': '100%'})


@app.callback(
    Output('right-panel-content', 'children'),
    Output('right-panel-header-text', 'children'),
    Output('selected-cluster', 'data'),
    Input('galaxy-3d', 'clickData'),
    Input({'type': 'lb-row', 'index': ALL}, 'n_clicks'),
    Input({'type': 'retro-row', 'index': ALL}, 'n_clicks'),
    Input({'type': 'back-btn', 'index': ALL}, 'n_clicks'),
    Input('horizon-sel', 'value'),
    Input('filter-ids', 'data'),
    Input('lb-direction', 'value'),
    Input('right-panel-header-text', 'n_clicks'),
    State('selected-cluster', 'data'),
    prevent_initial_call=True,
)
def update_right_panel(click_data, lb_clicks, retro_clicks, back_clicks, horizon,
                       filter_ids_list, direction, home_clicks, cur_sel):
    triggered = ctx.triggered_id
    fids = set(filter_ids_list) if filter_ids_list is not None else None

    def default_view():
        """The ranking view: filtered area (with grower/decliner chart) or full leaderboard."""
        if fids is not None:
            n = len(fids)
            return (build_filtered_list(fids, horizon, direction),
                    f'{n} CLUSTER{"S" if n != 1 else ""} MATCHED', None)
        title = 'TOP DECLINERS' if direction == 'decliners' else 'EMERGING CLUSTERS'
        return build_leaderboard(horizon, direction), title, None

    # Home click on the header → always return to the ranking
    if triggered == 'right-panel-header-text':
        return default_view()

    # Grower/decliner toggle → re-render ranking (drops any open detail)
    if triggered == 'lb-direction':
        return default_view()

    # Back button → ranking
    if isinstance(triggered, dict) and triggered.get('type') == 'back-btn' and any(back_clicks or []):
        return default_view()

    # Filter changed → keep detail if one is open, else ranking
    if triggered == 'filter-ids':
        if cur_sel is not None:
            return build_detail(cur_sel, horizon), _detail_header(cur_sel), cur_sel
        return default_view()

    # Horizon change → refresh current view
    if triggered == 'horizon-sel':
        if cur_sel is not None:
            return build_detail(cur_sel, horizon), _detail_header(cur_sel), cur_sel
        return default_view()

    # Retrospective row click
    if isinstance(triggered, dict) and triggered.get('type') == 'retro-row':
        cid = triggered['index']
        return build_detail(cid, horizon), _detail_header(cid), cid

    # Leaderboard / filtered-list row click
    if isinstance(triggered, dict) and triggered.get('type') == 'lb-row':
        cid = triggered['index']
        return build_detail(cid, horizon), _detail_header(cid), cid

    # Galaxy node click
    if triggered == 'galaxy-3d' and click_data:
        pt  = click_data['points'][0]
        cid = pt.get('customdata')
        if cid is not None:
            cid = int(cid)
            return build_detail(cid, horizon), _detail_header(cid), cid

    raise dash.exceptions.PreventUpdate


def _detail_header(cid):
    lbl = D['labels'].get(int(cid), f'Cluster {cid}')
    return html.Span([
        html.Span('CLUSTER', style={'color': MUTED, 'marginRight': '8px'}),
        html.Span(lbl[:30], style={'color': TEXT}),
    ])



@app.callback(
    Output('intel-content', 'children'),
    Input('intel-tabs', 'value'),
)
def update_intel(tab):
    if tab == 'insights':
        rows = _analytics_rows()
        return html.Div([
            html.Div([
                html.Span('▸', style={'color': NEON, 'marginRight': '8px', 'flexShrink': '0'}),
                html.Span(m, style={'width': '210px', 'flexShrink': '0', 'color': TEXT,
                                     'fontSize': '11px', 'fontWeight': '600'}),
                html.Span(v, style={'width': '230px', 'flexShrink': '0', 'color': NEON,
                                     'fontSize': '11px', 'fontFamily': 'Space Grotesk, monospace'}),
                html.Span(why, style={'flex': '1', 'color': '#94a3b8', 'fontSize': '10px',
                                       'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'padding': '4px 0',
                      'borderBottom': ' 1px solid rgba(255,255,255,0.04)'})
            for m, v, why in rows
        ])

    if tab == 'investor':
        if COMPANIES is None:
            return html.P('Company data not available.', style={'color': MUTED, 'fontSize': '12px'})
        top = COMPANIES.sort_values('exposure', ascending=False).head(10)
        mx  = float(top['exposure'].max()) or 1
        rows = []
        for i, r in enumerate(top.itertuples(), 1):
            w = r.exposure / mx * 100
            rows.append(html.Div([
                html.Span(f'{i}', style={'color': '#334155', 'fontSize': '10px', 'width': '16px',
                                          'flexShrink': '0', 'fontFamily': 'monospace'}),
                html.Span([
                    *([html.Span('⚡', style={'color': '#fbbf24', 'marginRight': '4px', 'fontSize': '9px'})]
                      if r.accelerating else []),
                    str(r.org)[:34],
                ], style={'width': '230px', 'flexShrink': '0', 'color': TEXT, 'fontSize': '11px',
                          'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                html.Span(f'{r.n_clusters} themes', style={'width': '64px', 'flexShrink': '0',
                          'color': MUTED, 'fontSize': '10px'}),
                html.Div(html.Div(style={'height': '3px', 'width': f'{w:.0f}%', 'background': NEON,
                         'borderRadius': '2px', 'boxShadow': f'0 0 5px {NEON}55'}),
                         style={'width': '90px', 'flexShrink': '0'}),
                html.Span(str(r.top_theme)[:24], style={'flex': '1', 'color': '#94a3b8', 'fontSize': '10px',
                          'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap',
                          'marginLeft': '8px'}),
            ], id={'type': 'company-row', 'index': str(r.org)}, className='company-row',
               n_clicks=0, title=f'Click to light up {r.org}’s {r.n_clusters} clusters in the galaxy'))
        return html.Div([
            html.Div('TOP COMPANIES BY EMERGING-TECH EXPOSURE   ·   click a company to map it · ⛶ expand for more',
                     style={'color': '#475569', 'fontSize': '9px', 'letterSpacing': '1px', 'marginBottom': '5px'}),
            *rows,
        ])

    if tab == 'overview':
        items = [
            ('Embed',    'all-MiniLM-L6-v2 sentence-transformers → 384-d vectors per patent title'),
            ('Cluster',  'UMAP(384→10d, cosine) + HDBSCAN(min=50) on rolling 2-yr windows'),
            ('Stitch',   'Cross-window centroid cosine matching (threshold 0.85) → global cluster IDs'),
            ('Features', '18 features: log_count, velocity, acceleration, jerk, momentum 4Q/8Q, MACD, z-score, rank pct, …'),
            ('Target',   'y = log_count[t+H] − log_count[t]  (growth delta in log-space, not absolute level)'),
            ('Model',    'GRU + LSTM  Input(8Q, 18F) → RNN(64) → Dropout → RNN(32) → Dense(1)  |  Huber loss  |  Spearman metric'),
            ('Forecast', 'Best model per horizon scores all clusters → ranked leaderboard'),
        ]
        return html.Div([
            html.Div([
                html.Span(k, style={'color': NEON, 'fontSize': '9px', 'letterSpacing': '1px',
                                     'fontWeight': '600', 'width': '64px', 'flexShrink': '0'}),
                html.Span('→', style={'color': MUTED, 'marginRight': '8px'}),
                html.Span(v, style={'color': '#94a3b8', 'fontSize': '11px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'padding': '5px 0',
                      'borderBottom': f'1px solid rgba(255,255,255,0.04)'})
            for k, v in items
        ])

    if tab == 'metrics':
        def hcell(t, w, align='left', color=MUTED):
            return html.Span(t, style={'width': w, 'flexShrink': '0', 'color': color,
                                       'fontSize': '9px', 'letterSpacing': '1px',
                                       'textAlign': align, 'fontWeight': '600'})
        def vcell(t, w, align='right', color=TEXT, bold=False):
            return html.Span(t, style={'width': w, 'flexShrink': '0', 'color': color,
                                       'fontSize': '11px', 'fontFamily': 'Space Grotesk, monospace',
                                       'textAlign': align, 'fontWeight': '700' if bold else '400'})

        blocks = []
        for hz in ('2yr', '3yr'):
            best = BEST_MODEL[hz]
            tm   = (TEST_METRICS or {}).get(hz, {})
            header = html.Div([
                hcell('MODEL', '108px'),
                hcell('VAL ρ', '52px', 'right'),
                hcell('TEST ρ', '56px', 'right'),
                hcell('RMSE', '52px', 'right'),
                hcell('MAE', '48px', 'right'),
            ], style={'display': 'flex', 'gap': '6px', 'padding': '4px 0',
                      'borderBottom': f'1px solid rgba(255,255,255,0.10)'})

            rows = []
            for model, m in VAL_METRICS[hz].items():
                is_best = (model == best)
                is_base = model in ('Linear trend', 'Persistence (0)')
                name_color = NEON if is_best else ('#64748b' if is_base else TEXT)
                sp_color   = NEON if m['spearman'] > 0.3 else (VIOLET if m['spearman'] < 0 else MUTED)
                # test ρ only for the best model (the one evaluated on test)
                test_sp = '—'
                if is_best and tm:
                    test_sp = f"{tm['test']['spearman']:+.3f}"
                rows.append(html.Div([
                    html.Span(('★ ' if is_best else '') + model,
                              style={'width': '108px', 'flexShrink': '0', 'color': name_color,
                                     'fontSize': '11px', 'fontWeight': '600' if is_best else '400'}),
                    vcell(f"{m['spearman']:+.3f}", '52px', color=sp_color, bold=is_best),
                    vcell(test_sp, '56px', color=NEON if test_sp != '—' else MUTED, bold=True),
                    vcell(f"{m['rmse']:.3f}", '52px', color='#94a3b8'),
                    vcell(f"{m['mae']:.3f}", '48px', color='#94a3b8'),
                ], style={'display': 'flex', 'gap': '6px', 'padding': '4px 0',
                          'borderBottom': f'1px solid rgba(255,255,255,0.04)'}))

            blocks.append(html.Div([
                html.Div(f'{hz.upper()} HORIZON  ({8 if hz=="2yr" else 12}Q)',
                         style={'color': '#475569', 'fontSize': '9px', 'letterSpacing': '2px',
                                'marginBottom': '4px', 'marginTop': '2px'}),
                header, *rows,
            ], style={'flex': '1', 'minWidth': '320px'}))

        note_test = ('Test ρ shown for ★ best model only — locked set, evaluated once.'
                     if TEST_METRICS else
                     'Test set still LOCKED — run scripts/eval_test.py (or Kaggle) to fill TEST ρ.')
        note = html.Div([
            html.Span('ρ = Spearman rank correlation.  ', style={'color': MUTED}),
            html.Span('LSTM beats linear-trend baseline (which is negative — naive momentum mis-ranks growth).  ',
                      style={'color': '#94a3b8'}),
            html.Span(note_test, style={'color': AMBER if not TEST_METRICS else NEON}),
        ], style={'fontSize': '10px', 'lineHeight': '1.5', 'marginTop': '8px',
                  'paddingTop': '8px', 'borderTop': f'1px solid rgba(255,255,255,0.06)'})

        return html.Div([
            html.Div(blocks, style={'display': 'flex', 'gap': '28px', 'flexWrap': 'wrap'}),
            note,
        ])

    if tab == 'data':
        panel = D['panel']
        n_cl = len(D['labels']) or len(D['coords3d'])
        items = [
            ('Patents indexed', f"{int(panel['count'].sum()):,}" if panel is not None else '—'),
            ('Clusters active', str(n_cl)),
            ('Quarters in panel', str(len(panel['date'].unique())) if panel is not None else '—'),
            ('Year range', f"{panel['date'].dt.year.min()} – {panel['date'].dt.year.max()}" if panel is not None else '—'),
            ('Train / Val / Test', '2005–2017  /  2018–2020  /  2021–2023 (locked)'),
            ('Embedding model', 'sentence-transformers/all-MiniLM-L6-v2'),
        ]
        return html.Div([
            html.Div([
                html.Span(k, style={'color': MUTED, 'fontSize': '10px', 'letterSpacing': '1px',
                                     'width': '160px', 'flexShrink': '0'}),
                html.Span(v, style={'color': TEXT, 'fontFamily': 'monospace', 'fontSize': '11px'}),
            ], style={'display': 'flex', 'padding': '5px 0',
                      'borderBottom': f'1px solid rgba(255,255,255,0.04)'})
            for k, v in items
        ])

    if tab == 'retrospective':
        retro_path = CDIR / 'retrospective_matches.csv'
        if not retro_path.exists():
            return html.Div([
                html.P('Retrospective matches not yet computed.', style={'color': MUTED, 'fontSize': '12px'}),
                html.P([
                    'Run: ',
                    html.Code('python scripts/match_eruptions.py',
                              style={'background': 'rgba(255,255,255,0.07)', 'padding': '2px 6px',
                                     'borderRadius': '4px', 'fontSize': '11px', 'color': NEON}),
                    ' (~5 min, requires pipeline_output.zip to be unzipped first)',
                ], style={'color': '#334155', 'fontSize': '11px', 'marginTop': '6px'}),
            ], style={'padding': '8px 0'})

        df = pd.read_csv(retro_path)
        conf = df[df['confident']].copy()
        conf['lead_str'] = conf['lead_time_q'].apply(
            lambda q: f'{int(q)}Q ({q/4:.1f}yr)' if pd.notna(q) and q > 0 else '—'
        )

        # Summary stats
        with_lead = conf[conf['lead_time_q'].notna() & (conf['lead_time_q'] > 0)]
        med_q = with_lead['lead_time_q'].median() if not with_lead.empty else None
        summary = html.Div([
            html.Span(f'{len(conf)} confident matches', style={'color': TEXT, 'fontSize': '12px', 'marginRight': '20px'}),
            *([] if med_q is None else [
                html.Span('Median lead time: ', style={'color': MUTED, 'fontSize': '12px'}),
                html.Span(f'{med_q:.0f}Q  ({med_q/4:.1f} yrs)',
                          style={'color': NEON, 'fontSize': '12px', 'fontFamily': 'monospace',
                                 'textShadow': f'0 0 8px rgba(0,212,255,0.4)'}),
            ]),
        ], style={'marginBottom': '8px'})

        # Table rows
        top = conf.sort_values('cosine_sim', ascending=False).head(20)
        header = html.Div([
            html.Span('TOPIC', style={'flex':'2','fontSize':'9px','color':MUTED,'letterSpacing':'1px'}),
            html.Span('MATCHED CLUSTER', style={'flex':'2','fontSize':'9px','color':MUTED,'letterSpacing':'1px'}),
            html.Span('SIM', style={'width':'36px','fontSize':'9px','color':MUTED,'letterSpacing':'1px','textAlign':'right'}),
            html.Span('LEAD', style={'width':'70px','fontSize':'9px','color':MUTED,'letterSpacing':'1px','textAlign':'right'}),
            html.Span('ERUPT', style={'width':'40px','fontSize':'9px','color':MUTED,'letterSpacing':'1px','textAlign':'right'}),
        ], style={'display':'flex','gap':'8px','padding':'4px 0','borderBottom':f'1px solid {BORDER}'})

        rows = []
        for _, r in top.iterrows():
            lead_c = NEON if (r.get('lead_time_q') or 0) > 4 else AMBER
            rows.append(html.Div([
                html.Span(str(r['topic_name'])[:28]+'…' if len(str(r['topic_name']))>28 else str(r['topic_name']),
                          style={'flex':'2','fontSize':'11px','color':TEXT,'overflow':'hidden',
                                 'whiteSpace':'nowrap','textOverflow':'ellipsis'}),
                html.Span(str(r['auto_label'])[:28]+'…' if len(str(r['auto_label']))>28 else str(r['auto_label']),
                          style={'flex':'2','fontSize':'11px','color':MUTED,'overflow':'hidden',
                                 'whiteSpace':'nowrap','textOverflow':'ellipsis'}),
                html.Span(f"{r['cosine_sim']:.2f}",
                          style={'width':'36px','fontSize':'11px','color':'#10b981','textAlign':'right','fontFamily':'monospace'}),
                html.Span(str(r['lead_str']),
                          style={'width':'70px','fontSize':'11px','color':lead_c,'textAlign':'right','fontFamily':'monospace'}),
                html.Span(str(int(r['eruption_year'])),
                          style={'width':'40px','fontSize':'11px','color':MUTED,'textAlign':'right','fontFamily':'monospace'}),
            ], id={'type': 'retro-row', 'index': int(r['cluster_id'])},
               className='lb-row', n_clicks=0))

        return html.Div([summary, header, html.Div(rows)], style={'height': '100%'})

    return html.Div()


@app.callback(
    Output('modal-container', 'children'),
    Output('modal-open', 'data'),
    Input('modal-open-btn', 'n_clicks'),
    Input({'type': 'modal-close', 'index': ALL}, 'n_clicks'),
    State('modal-open', 'data'),
    prevent_initial_call=True,
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if ctx.triggered_id == 'modal-open-btn':
        return build_modal(), True
    return [], False


@app.callback(
    Output('tab-modal-container', 'children'),
    Input('tab-expand-btn', 'n_clicks'),
    Input({'type': 'tab-modal-close', 'index': ALL}, 'n_clicks'),
    State('intel-tabs', 'value'),
    prevent_initial_call=True,
)
def toggle_tab_modal(expand_clicks, close_clicks, active_tab):
    if ctx.triggered_id == 'tab-expand-btn':
        return build_tab_modal(active_tab)
    return []


# ── Collapse / expand the bottom intelligence strip ─────────────────────────────
@app.callback(
    Output('intel-strip', 'style'),
    Output('main-area', 'style'),
    Output('intel-content', 'style'),
    Output('intel-collapse-btn', 'children'),
    Input('intel-collapse-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def toggle_intel(n):
    collapsed = (n or 0) % 2 == 1
    strip_p, main_p, content_p = Patch(), Patch(), Patch()
    if collapsed:
        strip_p['height'] = '40px'          # just the tab bar
        main_p['bottom']  = '40px'          # galaxy reclaims the space
        content_p['display'] = 'none'
        return strip_p, main_p, content_p, '▴'
    strip_p['height'] = '185px'
    main_p['bottom']  = '185px'
    content_p['display'] = 'block'
    return strip_p, main_p, content_p, '▾'


@app.callback(
    Output('watchlist-download', 'data'),
    Input('export-btn', 'n_clicks'),
    State('horizon-sel', 'value'),
    prevent_initial_call=True,
)
def export_watchlist(n, horizon):
    """Download the ranked forecast as a decision-ready CSV (top 50 by conviction)."""
    if CONV_DF is None:
        raise dash.exceptions.PreventUpdate
    df = CONV_DF.copy()
    df['domain']        = df['cluster_id'].map(lambda c: DOMAINS.get(int(c), 'Other'))
    df['accelerating']  = df['cluster_id'].map(lambda c: 'yes' if is_accelerating(c) else '')
    df['x_fold_2yr']    = np.exp(df['growth_2yr']).round(1)
    df['x_fold_3yr']    = np.exp(df['growth_3yr']).round(1)
    # top sample patent link per cluster
    def top_patent(c):
        ts = D.get('titles_ids', {}).get(int(c)) or []
        for pid, _ in ts:
            if pid:
                return f'https://patents.google.com/patent/US{pid}'
        return ''
    df['top_patent'] = df['cluster_id'].map(top_patent)
    def top_filers(c):
        orgs = D.get('assignees', {}).get(int(c)) or []
        return '; '.join(o for o, _ in orgs[:3])
    df['top_filers'] = df['cluster_id'].map(top_filers)
    out = (df.sort_values('conviction', ascending=False)
             .head(50)
             [['cluster_id', 'auto_label', 'domain', 'conviction', 'growth_2yr',
               'x_fold_2yr', 'growth_3yr', 'x_fold_3yr', 'volume', 'accelerating',
               'top_filers', 'top_patent']]
             .rename(columns={'auto_label': 'technology_cluster'}))
    return dcc.send_data_frame(out.to_csv, 'patent_radar_watchlist.csv', index=False)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
