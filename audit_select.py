import numpy as np,pandas as pd,json
from pathlib import Path
from collections import Counter
from recommend import read_archive,WEIGHTS,pndcg
from audit_experiments import gains
tr,va,te,mo,truth=read_archive(Path('data/selection_formula.zip'))
_,_,_,_,y,_,_,_=pd.read_pickle('work/audit_full_v5.pkl')
a=np.load('work/audit_scores_mixed_full_soft.npz');b=np.load('work/audit_scores_mixed_content.npz')
results=[];best=None
for w in [0,.25,.5,.75,1]:
 pred=(1-w)*a['scores']+w*b['scores'];score=gains(pred,y)
 item={'content_weight':w,'dev':float(score[a['dev']].mean()),'audit':float(score[a['audit']].mean())}
 results.append(item)
 if best is None or item['dev']>best[0]:best=(item['dev'],w,score,pred)
score=best[2];selected={'weight_by_dev_only':best[1],'dev':best[0],'audit':float(score[a['audit']].mean())}
counts=Counter()
for row in tr:
 for h in row['history']:
  if h['period']==2:
   for c in h['choices']:counts[c['module_id']]+=WEIGHTS[c['priority']]
avail=va[0]['available_modules'];coldrec=sorted(avail,key=lambda m:(-counts[m],m))[:5]
coldgain=np.array([pndcg({r['id']:coldrec},{r['id']:truth[r['id']]}) for r in va])
adjusted=np.where(a['cold'],coldgain,score)
selected['cold_prior_dev']=float(adjusted[a['dev']].mean());selected['cold_prior_audit']=float(adjusted[a['audit']].mean());selected['cold_prior_all_cold']=float(coldgain[a['cold']].mean());selected['cold_prior_audit_cold']=float(coldgain[a['cold']&a['audit']].mean());selected['model_audit_cold']=float(score[a['cold']&a['audit']].mean())
original=json.loads(Path('audit/original_predictions.json').read_text());orig=np.array([pndcg({r['id']:original[r['id']]},{r['id']:truth[r['id']]}) for r in va if r['id'] in original]);new=adjusted[a['audit']];delta=new-orig
rng=np.random.default_rng(991);boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)])
selected['delta_vs_original']=float(delta.mean());selected['paired_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
Path('audit/ensemble_comparison.json').write_text(json.dumps({'experiments':results,'selected':selected},indent=2));np.savez_compressed('work/selected_audit.npz',pred=best[3],score=adjusted,dev=a['dev'],audit=a['audit'],cold=a['cold'])
print(json.dumps({'experiments':results,'selected':selected},indent=2))
