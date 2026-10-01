"""Independent checks of the supplied archive, metric, and causal feature cutoff."""
import csv, hashlib, importlib.util, json, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from recommend import read_archive,choice_events,FeatureMaker,pndcg

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('official',ROOT/'vendor/metric.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
train,val,test,modules,truth=read_archive(ROOT/'data/selection_formula.zip')
known={m['module_id']:m for m in modules}
checks={}
for name,rows,key in [('train',train,'student_id'),('val',val,'id'),('test',test,'id')]:
 assert len(rows)==len({r[key] for r in rows})
 for r in rows:
  assert len(r['history'])==len({e['period'] for e in r['history']})
  for e in r['history']:
   assert len(e['choices'])==5
   assert len({c['module_id'] for c in e['choices']})==5
   assert {c['priority'] for c in e['choices']}=={1,2,3,4,5}
   assert all(e['period'] in known[c['module_id']]['available_in_periods'] for c in e['choices'])
  if 'available_modules' in r:
   assert len(r['available_modules'])==len(set(r['available_modules']))
   assert set(r['available_modules'])<=set(known)
   assert set(r['available_modules'])=={m for m in known if (3 if name=='val' else 4) in known[m]['available_in_periods']}
 checks[name]={'rows':len(rows),'cold':sum(not r['history'] for r in rows),'history_counts':dict(Counter(len(r['history']) for r in rows))}
assert set(truth)=={r['id'] for r in val}
for r in val:assert set(truth[r['id']])<=set(r['available_modules'])
observed={}
for rows,key in [(train,'student_id'),(val,'id'),(test,'id')]:
 for r in rows:
  for e in r['history']:
   k=(r[key],e['period']); ch={c['module_id']:c['priority'] for c in e['choices']}
   if k in observed:assert observed[k]==ch
   observed[k]=ch
for r in test:
 for e in r['history']:
  if e['period']==3:assert {c['module_id']:c['priority'] for c in e['choices']}==truth[r['id']]
checks['history_consistency']='all shared student/period records match; test period3 matches val labels'
# Official implementation, independently imported, agrees on edge cases and random lists.
rng=np.random.default_rng(971)
for _ in range(1000):
 gt=dict(zip(rng.choice(list(known),5,replace=False),range(1,6)))
 rec=rng.choice(list(known),5,replace=False).tolist()
 assert abs(pndcg({'x':rec},{'x':gt})-official.pndcg_single(gt,rec))<1e-12
assert official.pndcg_single({'m':1},['m'])==1
assert official.pndcg_single({'m':2},['m'])==.8
checks['metric']='official cap-at-one weighted DCG; 1000 randomized comparisons passed'
# Target-period labels cannot affect target-period features.
events=choice_events(train,val,truth)
maker=FeatureMaker(modules,events)
sample=[(r,None) for r in val[:8]]
x,_,_=maker.frame(sample,3)
mutated={k:(p,{m:6-v for m,v in a.items()}) if k[1]==3 else (p,a) for k,(p,a) in events.items()}
x2,_,_=FeatureMaker(modules,mutated).frame(sample,3)
assert x.equals(x2)
checks['label_cutoff']='period3 label permutation leaves period3 features unchanged'
checks['cold_fraction']={'val':checks['val']['cold']/len(val),'test':checks['test']['cold']/len(test)}
checks['new_modules']={str(p):sum(min(m['available_in_periods'])==p for m in modules) for p in (2,3,4)}
# Recompute exposure denominators independently for representative modules.
probe=val[0];frame,_,_=maker.frame([(probe,None)],3)
for module in ['m_002','m_279','m_092','m_179']:
 if module not in set(frame['module']):continue
 numerator=0.;denominator=0
 for row in train:
  for h in row['history']:
   if h['period'] in known[module]['available_in_periods']:
    denominator+=1
    numerator+=sum(official.WEIGHTS[c['priority']] for c in h['choices'] if c['module_id']==module)
 actual=float(frame.loc[frame['module']==module,'global_rate'].iloc[0])
 np.testing.assert_allclose(actual,numerator/max(denominator,1),rtol=1e-5,atol=1e-8)  # float32 accumulation
checks['exposure_rates']='four representative module denominators independently recomputed'
checks['archive_sha256']=hashlib.sha256((ROOT/'data/selection_formula.zip').read_bytes()).hexdigest()
checks['official_metric_sha256']=hashlib.sha256((ROOT/'vendor/metric.py').read_bytes()).hexdigest()
(ROOT/'audit/data_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
