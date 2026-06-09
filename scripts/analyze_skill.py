"""
analyze_skill.py — where does the model's ranking skill live?

Tests the claim that forecast skill is concentrated at the extremes of the
ranking (top growers / bottom decliners) and near-zero in the muddy middle.

Uses the VALIDATION set (development diagnostics — does not touch the locked test).

Outputs:
  - overall Spearman
  - decile-lift table: bin val samples by PREDICTED growth into 10 deciles,
    show mean predicted vs mean ACTUAL in each → the model should separate the
    extremes far more than the middle
  - precision@top / @bottom vs base rate
  - Spearman within top-10% / middle-80% / bottom-10% (with range caveat)

Run:  python scripts/analyze_skill.py
"""
import pathlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
import joblib

BASE = pathlib.Path(__file__).parent.parent
CDIR = BASE / 'data' / 'processed' / 'clusters'
MDIR = CDIR / 'models'

WINDOW = 8
TRAIN_END = (2017, 4)
VAL_END   = (2020, 4)
ALL_FEATURES = [
    'log_count','count_share','velocity','acceleration','jerk','mom_4q','mom_8q',
    'roll_mean_4q','roll_std_4q','roll_mean_8q','roll_std_8q','above_trend','macd',
    'z_score','global_rank_pctl','log_cumsum','sin_q','cos_q']
TARGET='log_count'; T,F = WINDOW, len(ALL_FEATURES)


def make_features(panel):
    p = panel.copy().sort_values(['cluster_id','date'])
    p['log_count']=np.log1p(p['count'])
    gq=p.groupby('date')['count'].transform('sum').replace(0,np.nan); p['count_share']=p['count']/gq
    g=p.groupby('cluster_id')['log_count']
    p['velocity']=g.diff().fillna(0); p['acceleration']=p.groupby('cluster_id')['velocity'].diff().fillna(0)
    p['jerk']=p.groupby('cluster_id')['acceleration'].diff().fillna(0)
    p['mom_4q']=p.groupby('cluster_id')['log_count'].diff(4).fillna(0)
    p['mom_8q']=p.groupby('cluster_id')['log_count'].diff(8).fillna(0)
    def roll(s,w,fn): return s.groupby(p['cluster_id']).transform(lambda x:getattr(x.rolling(w,min_periods=1),fn)())
    p['roll_mean_4q']=roll(p['log_count'],4,'mean'); p['roll_std_4q']=roll(p['log_count'],4,'std').fillna(0)
    p['roll_mean_8q']=roll(p['log_count'],8,'mean'); p['roll_std_8q']=roll(p['log_count'],8,'std').fillna(0)
    p['above_trend']=p['log_count']-p['roll_mean_8q']; p['macd']=p['roll_mean_4q']-p['roll_mean_8q']
    p['z_score']=p['above_trend']/(p['roll_std_8q']+1e-6)
    p['global_rank_pctl']=p.groupby('date')['log_count'].rank(pct=True)
    p['log_cumsum']=np.log1p(p.groupby('cluster_id')['count'].cumsum())
    p['sin_q']=np.sin(2*np.pi*p['quarter']/4); p['cos_q']=np.cos(2*np.pi*p['quarter']/4)
    return p.fillna(0)


def build_val(df, H):
    Xs,ys=[],[]
    for cid,grp in df.groupby('cluster_id'):
        grp=grp.sort_values('date').reset_index(drop=True); vals=grp[ALL_FEATURES].values; n=len(vals)
        for i in range(WINDOW,n-H+1):
            ar=grp.iloc[i-1]; ay,aq=int(ar['year']),int(ar['quarter'])
            in_train=(ay,aq)<=TRAIN_END; in_val=(not in_train) and (ay,aq)<=VAL_END
            if not in_val: continue
            ys.append(float(grp.iloc[i+H-1][TARGET])-float(grp.iloc[i-1][TARGET])); Xs.append(vals[i-WINDOW:i])
    return np.array(Xs,np.float32), np.array(ys,np.float32)


def main():
    from tensorflow import keras
    panel=pd.read_csv(CDIR/'cluster_panel.csv'); df=make_features(panel)

    for H in (8,12):
        hl=f'{H//4}yr'
        X,y=build_val(df,H)
        sc=joblib.load(MDIR/f'scaler_{hl}.pkl')
        Xs=sc.transform(X.reshape(-1,F)).reshape(-1,T,F)
        m=sorted(MDIR.glob(f'lstm_{hl}_*.keras'))[-1]
        pred=keras.models.load_model(m).predict(Xs,verbose=0).flatten()

        d=pd.DataFrame({'pred':pred,'actual':y})
        n=len(d)
        overall=spearmanr(d['pred'],d['actual'])[0]
        print(f'\n{"="*64}\nHORIZON {hl}  —  {n:,} validation samples\n{"="*64}')
        print(f'Overall Spearman ρ = {overall:.3f}')

        # decile lift by PREDICTED value
        d['decile']=pd.qcut(d['pred'].rank(method='first'),10,labels=False)+1
        print('\nDECILE LIFT (binned by predicted growth):')
        print(f"  {'decile':>6} {'mean_pred':>10} {'mean_actual':>12} {'n':>6}")
        tab=d.groupby('decile').agg(mean_pred=('pred','mean'),mean_actual=('actual','mean'),n=('actual','size'))
        for dec,r in tab.iterrows():
            bar='█'*max(0,int(abs(r.mean_actual)*8))
            print(f"  {dec:>6} {r.mean_pred:>10.3f} {r.mean_actual:>12.3f} {int(r.n):>6}  {bar}")
        top_act=tab.loc[10,'mean_actual']; bot_act=tab.loc[1,'mean_actual']; mid_act=tab.loc[5:6,'mean_actual'].mean()
        print(f'\n  Top decile actual:    {top_act:+.3f}')
        print(f'  Bottom decile actual: {bot_act:+.3f}')
        print(f'  Middle deciles actual:{mid_act:+.3f}  (should be near 0 / flat)')
        print(f'  Spread top−bottom:    {top_act-bot_act:.3f}')

        # precision @ extremes vs base rate
        base_up=float((d['actual']>0).mean())
        topN=d.nlargest(max(1,n//10),'pred'); botN=d.nsmallest(max(1,n//10),'pred')
        print(f'\n  Base rate P(actual grows) = {base_up:.0%}')
        print(f'  Of top-10% predicted: {float((topN.actual>0).mean()):.0%} actually grew')
        print(f'  Of bottom-10% predicted: {float((botN.actual<0).mean()):.0%} actually declined')

        # Spearman by region (range-restricted — caveat)
        order=d.sort_values('pred'); k=n//10
        bottom=order.iloc[:k]; middle=order.iloc[k:-k]; topp=order.iloc[-k:]
        sp_b=spearmanr(bottom.pred,bottom.actual)[0]
        sp_m=spearmanr(middle.pred,middle.actual)[0]
        sp_t=spearmanr(topp.pred,topp.actual)[0]
        print(f'\n  Spearman WITHIN bottom-10%: {sp_b:+.3f}')
        print(f'  Spearman WITHIN middle-80%: {sp_m:+.3f}   <- the muddy middle')
        print(f'  Spearman WITHIN top-10%:    {sp_t:+.3f}')
        print('  (within-band ρ is range-restricted; read the decile-lift table as the main evidence)')


if __name__ == '__main__':
    main()
