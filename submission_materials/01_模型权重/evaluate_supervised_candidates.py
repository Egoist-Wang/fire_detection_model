from __future__ import annotations
import csv, json
from pathlib import Path
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

BASE = Path(r'C:\fire_detection_clean')
ROOT = BASE/'recovery_20261017'/'sprint_20261017'
SCORES = BASE/'competition_20261014'/'score_cache'
def read_scores(names):
    cols=[]
    for k in ('yolo','siglip2','dinov3'):
        d={}
        with (SCORES/f'{k}.csv').open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f): d[r['image_name']]=float(r['score'])
        cols.append([d[n] for n in names])
    return np.asarray(cols,dtype=np.float32).T

def score_report(y, p):
    return {'f1':float(f1_score(y,p,zero_division=0)), 'precision':float(precision_score(y,p,zero_division=0)), 'recall':float(recall_score(y,p,zero_division=0)), 'positive':int(p.sum())}

def best_threshold(y, prob):
    best=None
    for t in np.linspace(.05,.95,361):
        p=(prob>=t).astype(int); z=score_report(y,p)
        key=(z['f1'],z['recall'])
        if best is None or key>(best[0],best[1]): best=(z['f1'],z['recall'],float(t),z)
    return best

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reviewed-json',type=Path,required=True)
    parser.add_argument('--original-json',type=Path,required=True)
    args=parser.parse_args()
    z=np.load(ROOT/'swin_features.npz',allow_pickle=False); names=[str(x) for x in z['names']]
    final=json.loads(args.reviewed_json.read_text(encoding='utf-8-sig')); y=np.asarray([final[n] for n in names],int)
    original=json.loads(args.original_json.read_text(encoding='utf-8-sig'))
    feat=z['features'].astype('float32'); scores=read_scores(names)
    # Compact feature variants. All transformations are unsupervised.
    s=np.clip(scores,1e-5,1-1e-5); slog=np.log(s/(1-s))
    f0=StandardScaler().fit_transform(feat)
    f64=np.linalg.svd(f0,full_matrices=False)[2][:64].T
    pc=f0@f64
    ss=StandardScaler().fit_transform(scores); sl=StandardScaler().fit_transform(slog)
    old=np.asarray([original[n] for n in names],np.float32)[:,None]
    variants={'scores':ss, 'scores_log':np.hstack([ss,sl]), 'pc_scores':np.hstack([pc,ss,sl]), 'pc_scores_nolog':np.hstack([pc,ss]), 'pc_scores_orig':np.hstack([pc,ss,old]), 'pc_scores_raw_orig':np.hstack([pc,scores,old])}
    a=np.arange(1999,2700); b=np.arange(3999,4500); known=np.r_[a,b]
    folds=[(a,b),(b,a)]
    models={
      'logistic':lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,class_weight='balanced',max_iter=2000)),
      'logistic_c1':lambda: make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000)),
      'extra':lambda: ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,max_features=.7,class_weight='balanced',random_state=7,n_jobs=-1),
      'rf':lambda: RandomForestClassifier(n_estimators=400,min_samples_leaf=3,max_features=.7,class_weight='balanced',random_state=7,n_jobs=-1),
      'hgb':lambda: HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,l2_regularization=3,learning_rate=.05,random_state=7),
      'svc':lambda: make_pipeline(StandardScaler(),SVC(C=1.0,class_weight='balanced',probability=True,random_state=7)),
      'knn':lambda: make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=31,weights='distance',p=2)),
    }
    allrep=[]
    for vn,X in variants.items():
      for mn,mk in models.items():
        oof=np.full(len(names),np.nan,float)
        for tr,va in folds:
          m=mk(); m.fit(X[tr],y[tr]);
          if hasattr(m,'predict_proba'): q=m.predict_proba(X[va])[:,1]
          else: q=m.decision_function(X[va]); q=(q-q.min())/(q.max()-q.min()+1e-9)
          oof[va]=q
        zbest=best_threshold(y[known],oof[known]); allrep.append({'variant':vn,'model':mn,'oof':zbest[3],'threshold':zbest[2]})
    allrep.sort(key=lambda r:(r['oof']['f1'],r['oof']['recall']),reverse=True)
    print(json.dumps(allrep[:20],ensure_ascii=False,indent=2))
    (ROOT/'supervised_cv_report.json').write_text(json.dumps(allrep,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
