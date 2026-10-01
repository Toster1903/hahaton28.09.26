from pathlib import Path
import numpy as np,pandas as pd,json
import importlib.util
spec=importlib.util.spec_from_file_location('old',Path(__file__).parent/'audit/previous_recommend.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
tr,va,te,mo,truth=old.read_archive(Path('data/selection_formula.zip'))
ids=np.array([r['id'] for r in va]);rng=np.random.default_rng(42);dev=set(rng.choice(ids,size=len(ids)//5,replace=False));rng=np.random.default_rng(20260928);aud=set(rng.choice([s for s in ids if s not in dev],size=600,replace=False))
fm=old.FeatureMaker(mo,old.choice_events(tr,va,truth))
p2=[(r,{c['module_id']:c['priority'] for h in r['history'] if h['period']==2 for c in h['choices']}) for r in tr if any(h['period']==2 for h in r['history'])]
x2,y2,_=fm.frame(p2,2,sample_negatives=True)
x3,y3,_=fm.frame([(r,truth[r['id']]) for r in va if r['id'] not in (dev|aud)],3,sample_negatives=True)
xd,yd,gd=fm.frame([(r,truth[r['id']]) for r in va if r['id'] in dev],3)
xa,ya,ga=fm.frame([(r,truth[r['id']]) for r in va if r['id'] in aud],3)
model=old.train_model(pd.concat([x2,x3],ignore_index=True),np.r_[y2,y3],iterations=450,eval_set=(xd,yd))
pred=old.ranked_predictions(model,xa,ga)
score=old.pndcg(pred,{s:truth[s] for s in aud})
res={'original_same_audit_score':score,'train_period3_students':len(va)-len(dev)-len(aud),'audit_n':600}
Path('audit/original_matched.json').write_text(json.dumps(res,indent=2))
Path('audit/original_predictions.json').write_text(json.dumps(pred))
print(json.dumps(res),flush=True)
