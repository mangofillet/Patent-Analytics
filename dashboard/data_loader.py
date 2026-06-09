import pathlib, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE  = pathlib.Path(__file__).parent.parent
CDIR  = BASE / 'data' / 'processed' / 'clusters'
CACHE = BASE / 'data' / 'processed'


def _csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def _pkl(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def compute_umap3d(centroids: dict) -> pd.DataFrame:
    empty = pd.DataFrame(columns=['cluster_id', 'x', 'y', 'z'])
    if not centroids:
        return empty

    cache = CACHE / 'umap3d_coords.csv'
    if cache.exists():
        return pd.read_csv(cache)

    try:
        import umap as umap_lib
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'umap-learn', '-q'], check=False)
        import umap as umap_lib

    # restrict to centroids that also appear in labels/panel (skip unlabeled intermediates)
    labels_csv = CDIR / 'cluster_labels.csv'
    if labels_csv.exists():
        active = set(pd.read_csv(labels_csv)['cluster_id'].tolist())
        ids = sorted(k for k in centroids if k in active)
    else:
        ids = sorted(centroids.keys())
    embs = np.array([centroids[i] for i in ids], dtype=np.float32)
    n_nb = min(15, max(2, len(ids) - 1))

    print(f'[data_loader] Computing UMAP-3D for {len(ids)} clusters …')
    red  = umap_lib.UMAP(n_components=3, metric='cosine', random_state=42,
                          n_neighbors=n_nb, min_dist=0.1)
    xyz  = red.fit_transform(embs)

    df = pd.DataFrame({'cluster_id': ids, 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]})
    df.to_csv(cache, index=False)
    print(f'[data_loader] UMAP-3D cached → {cache}')
    return df


DOMAIN_KEYWORDS = {
    'AI / ML':          ['neural', 'deep learning', 'machine learning', 'transformer',
                         'language model', 'inference', 'training', 'classifier', 'reinforcement'],
    'Biotech':          ['gene', 'protein', 'cell', 'dna', 'rna', 'cancer', 'antibody',
                         'genome', 'crispr', 'drug', 'therapeutic', 'peptide', 'enzyme',
                         'nucleic', 'antibody', 'immunotherapy'],
    'Semiconductors':   ['semiconductor', 'transistor', 'circuit', 'memory', 'processor',
                         'wafer', 'diode', 'integrated circuit', 'mosfet', 'chip fabrication',
                         'photolithography', 'silicon', 'substrate'],
    'Energy':           ['battery', 'solar', 'fuel cell', 'photovoltaic', 'hydrogen',
                         'wind turbine', 'energy storage', 'electrolyte', 'lithium',
                         'electrode', 'charging', 'power grid'],
    'Communications':   ['wireless', '5g', 'antenna', 'spectrum', 'radio', 'optical fiber',
                         'network protocol', 'base station', 'mimo', 'beamforming',
                         'bandwidth', 'modulation'],
    'Materials':        ['polymer', 'composite', 'alloy', 'graphene', 'nanotube',
                         'ceramic', 'coating', 'nanoparticle', 'thin film', 'catalyst'],
    'Robotics':         ['robot', 'actuator', 'autonomous', 'servo', 'motion control',
                         'end effector', 'lidar', 'path planning', 'manipulation', 'drone'],
    'Software / Security': ['encryption', 'cybersecurity', 'authentication', 'blockchain',
                             'algorithm', 'data structure', 'operating system', 'compiler',
                             'hash', 'key management'],
    'Medical Devices':  ['catheter', 'implant', 'stent', 'surgical', 'endoscope',
                         'prosthetic', 'imaging', 'ultrasound', 'cardiac', 'orthopedic'],
    'Display / Optics': ['display', 'led', 'oled', 'lcd', 'pixel', 'backlight',
                         'optical lens', 'waveguide', 'photonic', 'holographic'],
}


def assign_domains(labels: dict) -> dict:
    """Map cluster_id → best-matching domain string (or 'Other')."""
    result = {}
    for cid, lbl in labels.items():
        lbl_lower = lbl.lower()
        best_domain, best_score = 'Other', 0
        for domain, kws in DOMAIN_KEYWORDS.items():
            score = sum(1 for k in kws if k in lbl_lower)
            if score > best_score:
                best_score, best_domain = score, domain
        result[int(cid)] = best_domain
    return result


def enrich_panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy().sort_values(['cluster_id', 'date'])
    p['log_count'] = np.log1p(p['count'])
    p['date']      = pd.to_datetime(p['date'])
    p['velocity']  = p.groupby('cluster_id')['log_count'].diff().fillna(0)
    return p


def load_all() -> dict:
    panel     = _csv(CDIR / 'cluster_panel.csv')
    labels_df = _csv(CDIR / 'cluster_labels.csv')
    labels    = ({int(k): v for k, v in zip(labels_df['cluster_id'], labels_df['auto_label'])}
                 if labels_df is not None else {})
    centroids = {int(k): v for k, v in _pkl(CDIR / 'cluster_centroids.pkl').items()}
    titles    = {int(k): v for k, v in _pkl(CDIR / 'cluster_titles.pkl').items()}
    # Optional: (patent_id, title) tuples per cluster, from add_patent_ids.py
    titles_ids = {int(k): v for k, v in _pkl(CDIR / 'cluster_titles_ids.pkl').items()}
    # Optional: [(org, count), ...] top filers per cluster, from add_assignees.py
    assignees  = {int(k): v for k, v in _pkl(CDIR / 'cluster_assignees.pkl').items()}
    # Optional: [(org, count, [(pid, title), ...]), ...] per cluster — filer→patents
    filer_pats = {int(k): v for k, v in _pkl(CDIR / 'cluster_filer_patents.pkl').items()}
    lb2       = _csv(CDIR / 'leaderboard_2yr.csv')
    lb3       = _csv(CDIR / 'leaderboard_3yr.csv')
    lb5       = _csv(CDIR / 'leaderboard_5yr_extrap.csv')
    coords3d  = compute_umap3d(centroids)

    if panel is not None:
        panel = enrich_panel(panel)

    # Summarise what's available
    n_cl  = len(labels) or len(centroids)
    n_lb  = len(lb2) if lb2 is not None else 0
    print(f'[data_loader] {n_cl} clusters | {n_lb} leaderboard entries | '
          f'UMAP-3D: {len(coords3d)} nodes')

    domains = assign_domains(labels)

    return dict(
        panel=panel, labels=labels, centroids=centroids, titles=titles,
        titles_ids=titles_ids, assignees=assignees, filer_pats=filer_pats,
        coords3d=coords3d, lb2=lb2, lb3=lb3, lb5=lb5,
        domains=domains,
    )
