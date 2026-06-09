"""
eval_test.py — LOCKED TEST SET evaluation

Reconstructs the test-set samples from the saved cluster files, loads the
trained models + scalers from results.zip output, and reports Spearman / RMSE /
MAE on the held-out test anchors. This is the one-time final evaluation.

Run:
    python scripts/eval_test.py
"""

import pathlib, pickle, json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import joblib

BASE   = pathlib.Path(__file__).parent.parent
CDIR   = BASE / 'data' / 'processed' / 'clusters'
MDIR   = CDIR / 'models'
OUT    = CDIR / 'test_metrics.json'

# ── Config (must match kaggle_train.ipynb) ─────────────────────────────────────
WINDOW    = 8
HORIZONS  = [8, 12]
TRAIN_END = (2017, 4)
VAL_END   = (2020, 4)
TEST_END  = {8: (2023, 4), 12: (2022, 4)}

ALL_FEATURES = [
    'log_count', 'count_share',
    'velocity', 'acceleration', 'jerk',
    'mom_4q', 'mom_8q',
    'roll_mean_4q', 'roll_std_4q', 'roll_mean_8q', 'roll_std_8q',
    'above_trend', 'macd', 'z_score',
    'global_rank_pctl', 'log_cumsum', 'sin_q', 'cos_q',
]
TARGET = 'log_count'
T, F   = WINDOW, len(ALL_FEATURES)


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy().sort_values(['cluster_id', 'date'])
    p['log_count']   = np.log1p(p['count'])
    global_q         = p.groupby('date')['count'].transform('sum').replace(0, np.nan)
    p['count_share'] = p['count'] / global_q
    grp = p.groupby('cluster_id')['log_count']
    p['velocity']     = grp.diff().fillna(0)
    p['acceleration'] = p.groupby('cluster_id')['velocity'].diff().fillna(0)
    p['jerk']         = p.groupby('cluster_id')['acceleration'].diff().fillna(0)
    p['mom_4q'] = p.groupby('cluster_id')['log_count'].diff(4).fillna(0)
    p['mom_8q'] = p.groupby('cluster_id')['log_count'].diff(8).fillna(0)
    def roll(series, w, fn):
        return series.groupby(p['cluster_id']).transform(
            lambda x: getattr(x.rolling(w, min_periods=1), fn)())
    p['roll_mean_4q'] = roll(p['log_count'], 4, 'mean')
    p['roll_std_4q']  = roll(p['log_count'], 4, 'std').fillna(0)
    p['roll_mean_8q'] = roll(p['log_count'], 8, 'mean')
    p['roll_std_8q']  = roll(p['log_count'], 8, 'std').fillna(0)
    p['above_trend']  = p['log_count'] - p['roll_mean_8q']
    p['macd']         = p['roll_mean_4q'] - p['roll_mean_8q']
    p['z_score']      = p['above_trend'] / (p['roll_std_8q'] + 1e-6)
    p['global_rank_pctl'] = p.groupby('date')['log_count'].rank(pct=True)
    p['log_cumsum']   = np.log1p(p.groupby('cluster_id')['count'].cumsum())
    p['sin_q']        = np.sin(2 * np.pi * p['quarter'] / 4)
    p['cos_q']        = np.cos(2 * np.pi * p['quarter'] / 4)
    return p.fillna(0)


def build_samples(df, split, horizon):
    Xs, ys = [], []
    for cid, grp in df.groupby('cluster_id'):
        grp  = grp.sort_values('date').reset_index(drop=True)
        vals = grp[ALL_FEATURES].values
        n    = len(vals)
        for i in range(WINDOW, n - horizon + 1):
            ar = grp.iloc[i - 1]
            ay, aq = int(ar['year']), int(ar['quarter'])
            in_train = (ay, aq) <= TRAIN_END
            in_val   = (not in_train) and (ay, aq) <= VAL_END
            in_test  = not in_train and not in_val and (ay, aq) <= TEST_END[horizon]
            if split == 'train' and not in_train: continue
            if split == 'val'   and not in_val:   continue
            if split == 'test'  and not in_test:  continue
            ys.append(float(grp.iloc[i + horizon - 1][TARGET]) - float(grp.iloc[i - 1][TARGET]))
            Xs.append(vals[i - WINDOW:i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def find_model(arch, hlabel):
    cands = sorted(MDIR.glob(f'{arch}_{hlabel}_*.keras'))
    return cands[-1] if cands else None


def main():
    from tensorflow import keras

    print('Loading cluster panel + computing features …')
    panel = pd.read_csv(CDIR / 'cluster_panel.csv')
    df    = make_features(panel)

    # Which arch was best per horizon (from val metrics — LSTM won both)
    BEST_ARCH = {8: 'lstm', 12: 'lstm'}

    results = {}
    for H in HORIZONS:
        hlabel = f'{H//4}yr'
        print(f'\n{"="*55}\nHorizon {hlabel} ({H}Q) — TEST SET\n{"="*55}')

        X_te, y_te = build_samples(df, 'test', H)
        X_v,  y_v  = build_samples(df, 'val',  H)
        if len(y_te) == 0:
            print('  No test samples.'); continue

        sc = joblib.load(MDIR / f'scaler_{hlabel}.pkl')
        X_te_sc = sc.transform(X_te.reshape(-1, F)).reshape(-1, T, F)
        X_v_sc  = sc.transform(X_v.reshape(-1, F)).reshape(-1, T, F)

        arch  = BEST_ARCH[H]
        mpath = find_model(arch, hlabel)
        model = keras.models.load_model(mpath)
        print(f'  Best model: {arch.upper()}  ({mpath.name})')

        pred_te = model.predict(X_te_sc, verbose=0).flatten()
        pred_v  = model.predict(X_v_sc,  verbose=0).flatten()

        # baselines on test
        y_persist = np.zeros_like(y_te)
        lci   = ALL_FEATURES.index('log_count')
        steps = np.arange(WINDOW, dtype=np.float32)
        A     = np.vstack([steps, np.ones(WINDOW)]).T
        slopes = np.linalg.lstsq(A, X_te[:, :, lci].T, rcond=None)[0][0]
        y_trend = slopes * H

        def metr(y, p):
            sp = spearmanr(y, p)[0]
            return dict(spearman=round(float(sp), 4),
                        rmse=round(float(np.sqrt(mean_squared_error(y, p))), 4),
                        mae=round(float(mean_absolute_error(y, p)), 4))

        m_test  = metr(y_te, pred_te)
        m_val   = metr(y_v,  pred_v)
        m_trend = metr(y_te, y_trend)
        rmse_p  = round(float(np.sqrt(mean_squared_error(y_te, y_persist))), 4)

        print(f'  n_test={len(y_te)}  n_val={len(y_v)}')
        print(f'  {arch.upper():12s} TEST  Spearman={m_test["spearman"]:.4f}  '
              f'RMSE={m_test["rmse"]:.4f}  MAE={m_test["mae"]:.4f}')
        print(f'  {arch.upper():12s} VAL   Spearman={m_val["spearman"]:.4f}')
        print(f'  Linear trend TEST  Spearman={m_trend["spearman"]:.4f}  RMSE={m_trend["rmse"]:.4f}')
        print(f'  Persistence  TEST  Spearman=0.0000  RMSE={rmse_p:.4f}')

        results[hlabel] = dict(
            best_arch=arch.upper(),
            n_test=int(len(y_te)), n_val=int(len(y_v)),
            test=m_test, val=m_val,
            baseline_trend=m_trend,
            baseline_persistence=dict(spearman=0.0, rmse=rmse_p),
        )

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {OUT}')


if __name__ == '__main__':
    main()
