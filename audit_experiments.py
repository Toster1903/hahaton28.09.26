"""Reproducible temporal and student-holdout evaluation on all candidates."""
import argparse,json,time,hashlib,inspect
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from recommend import read_archive,choice_events,FeatureMaker,CATS
ROOT=Path(__file__).parent
CACHE=ROOT/'work/audit_full_v5.pkl'
META=CACHE.with_suffix('.meta.json')
def fingerprint():
 return {'archive':hashlib.sha256((ROOT/'data/selection_formula.zip').read_bytes()).hexdigest(),'features':hashlib.sha256(inspect.getsource(FeatureMaker).encode()).hexdigest()}
def cache_valid():
 return CACHE.exists() and META.exists() and json.loads(META.read_text())==fingerprint()

def prepare():
 tr,va,te,mo,truth=read_archive(ROOT/'data/selection_formula.zip')
 fm=FeatureMaker(mo,choice_events(tr,va,truth))
 rec2=[(r,{c['module_id']:c['priority'] for e in r['history'] if e['period']==2 for c in e['choices']}) for r in tr if any(e['period']==2 for e in r['history'])]
 x2,y2,g2=fm.frame(rec2,2)
 x3,y3,g3=fm.frame([(r,truth[r['id']]) for r in va],3)
 payload=(x2,y2,g2,x3,y3,g3,va,truth)
 pd.to_pickle(payload,CACHE)
 META.write_text(json.dumps(fingerprint(),indent=2))
 print('full candidate rows',len(x2),len(x3),flush=True)
 return payload

def gains(pred,y,n=159):
 score=pred.reshape(-1,n);target=y.reshape(-1,n)
 idx=np.argsort(-score,axis=1,kind='stable')[:,:5]
 return np.minimum(1,(np.take_along_axis(target,idx,axis=1)/np.log2(np.arange(2,7))).sum(axis=1))

def run(a):
 x2,y2,g2,x3,y3,g3,va,truth=pd.read_pickle(CACHE) if cache_valid() else prepare()
 ids=np.array([r['id'] for r in va]);rng=np.random.default_rng(42)
 dev=set(rng.choice(ids,size=len(ids)//5,replace=False))
 remaining=np.array([s for s in ids if s not in dev]);rng=np.random.default_rng(20260928)
 audit=set(rng.choice(remaining,size=600,replace=False))
 devmask=np.array([s in dev for s in g3]);auditmask=np.array([s in audit for s in g3]);fitmask=~(devmask|auditmask)
 cats=list(CATS)
 if a.history_categories:
  tr,_,_,_,_=read_archive(ROOT/'data/selection_formula.zip')
  def enrich(frame,groups,records,period):
   mapping={}
   for r in records:
    hist=[h for h in r['history'] if h['period']<period]
    latest=max(hist,key=lambda h:h['period'])['choices'] if hist else []
    by_priority={c['priority']:c['module_id'] for c in latest}
    mapping[r.get('id',r.get('student_id'))]=[by_priority.get(i,'none') for i in range(1,6)]
   for j in range(5):frame[f'previous_choice_{j+1}']=[mapping[s][j] for s in groups]
   frame['program']=frame['institute']+' | '+frame['specialty']
  enrich(x2,g2,tr,2);enrich(x3,g3,va,3)
  cats += [f'previous_choice_{i}' for i in range(1,6)]+['program']
 drop=a.drop.split(',') if a.drop else []
 x2=x2.drop(columns=drop);x3=x3.drop(columns=drop)
 for col in [c for c in cats if c not in drop]:
  categories=sorted(set(x2[col])|set(x3[col]));x2[col]=pd.Categorical(x2[col],categories);x3[col]=pd.Categorical(x3[col],categories)
 if a.mode=='mixed':
  x=pd.concat([x2,x3.loc[fitmask]],ignore_index=True); y=np.r_[y2,y3[fitmask]]
  weights=np.r_[np.ones(len(y2)),np.full(fitmask.sum(),a.period3_weight)]
 else:x=x2;y=y2;weights=np.ones(len(y2))
 if a.add_period1:
  p1path=ROOT/'work/audit_period1_v5.pkl'
  if p1path.exists():x1,y1=pd.read_pickle(p1path)
  else:
   tr,_,_,mo,truth1=read_archive(ROOT/'data/selection_formula.zip');fm=FeatureMaker(mo,choice_events(tr,va,truth1))
   records=[(r,{c['module_id']:c['priority'] for h in r['history'] if h['period']==1 for c in h['choices']}) for r in tr if any(h['period']==1 for h in r['history'])]
   x1,y1,_=fm.frame(records,1);pd.to_pickle((x1,y1),p1path)
  x1=x1.drop(columns=drop)
  for col in [c for c in cats if c not in drop]:
   categories=sorted(set(x1[col].astype(str))|set(x[col].astype(str))|set(x3[col].astype(str)))
   x1[col]=pd.Categorical(x1[col],categories);x[col]=pd.Categorical(x[col],categories);x3[col]=pd.Categorical(x3[col],categories)
  x=pd.concat([x1,x],ignore_index=True);y=np.r_[y1,y];weights=np.r_[np.full(len(y1),.33),weights]
 params=dict(n_estimators=a.trees,learning_rate=.035,num_leaves=a.leaves,min_child_samples=60,colsample_bytree=.9,reg_lambda=8,verbosity=-1,random_state=42,n_jobs=6,metric='None')
 def metric(ytrue,pred):return 'pndcg5',float(gains(pred,y3[devmask]).mean()),True
 if a.model=='binary':
  model=lgb.LGBMClassifier(**params)
  model.fit(x,(y>0).astype(int),sample_weight=weights*np.where(y>0,y,1),eval_set=[(x3.loc[devmask],(y3[devmask]>0).astype(int))],eval_metric=metric,callbacks=([lgb.log_evaluation(100)] if a.fixed_iterations else [lgb.early_stopping(65,verbose=False),lgb.log_evaluation(100)]))
  pred=model.predict_proba(x3)[:,1]
 else:
  model=lgb.LGBMRegressor(objective='cross_entropy' if a.model=='soft' else 'regression',**params)
  model.fit(x,y,sample_weight=weights,eval_set=[(x3.loc[devmask],y3[devmask])],eval_metric=metric,callbacks=([lgb.log_evaluation(100)] if a.fixed_iterations else [lgb.early_stopping(65,verbose=False),lgb.log_evaluation(100)]))
  pred=model.predict(x3)
 scores=gains(pred,y3);cold=np.array([not r['history'] for r in va]);devu=np.array([s in dev for s in ids]);auditu=np.array([s in audit for s in ids])
 results={'tag':a.tag,'mode':a.mode,'model':a.model,'drop':drop,'leaves':a.leaves,'best_iteration':model.best_iteration_ or a.trees,'dev':float(scores[devu].mean()),'audit':float(scores[auditu].mean()),'audit_cold':float(scores[auditu&cold].mean()),'audit_warm':float(scores[auditu&~cold].mean()),'audit_cold_n':int((auditu&cold).sum()),'dev_cold':float(scores[devu&cold].mean()),'dev_warm':float(scores[devu&~cold].mean())}
 if a.mode=='temporal':results['full_temporal_tuned']=float(scores.mean())
 print(json.dumps(results),flush=True)
 (ROOT/'audit'/f'{a.tag}.json').write_text(json.dumps(results,indent=2))
 np.savez_compressed(ROOT/'work'/f'audit_scores_{a.tag}.npz',scores=pred,ids=ids,dev=devu,audit=auditu,cold=cold)
 model.booster_.save_model(str(ROOT/'work'/f'{a.tag}.lgb'))
 return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['mixed','temporal'],default='mixed');p.add_argument('--model',choices=['binary','soft','regression'],default='binary');p.add_argument('--drop',default='');p.add_argument('--leaves',type=int,default=127);p.add_argument('--trees',type=int,default=650);p.add_argument('--period3-weight',type=float,default=3);p.add_argument('--history-categories',action='store_true');p.add_argument('--add-period1',action='store_true');p.add_argument('--fixed-iterations',action='store_true')
 run(p.parse_args())
