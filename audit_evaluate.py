"""Verify the selected ensemble with the unmodified official scoring implementation."""
from pathlib import Path
import csv,json,importlib.util
import numpy as np
from recommend import read_archive,choice_events,FeatureMaker,cold_scores,validate_submission
ROOT=Path(__file__).parent
tr,va,te,mo,truth=read_archive(ROOT/'data/selection_formula.zip')
fm=FeatureMaker(mo,choice_events(tr,va,truth));cold=cold_scores(fm,3)
a=np.load(ROOT/'work/selected_audit.npz');scores=a['pred'].reshape(len(va),-1);auditmask=a['audit']
selected=[r for r,keep in zip(va,auditmask) if keep];predictions={}
for row,score,keep in zip(va,scores,auditmask):
 if not keep:continue
 if not row['history'] and not any(row['profile'].values()):
  rec=sorted(row['available_modules'],key=lambda m:(-cold[fm.lookup[m]],m))[:5]
 else:
  rec=[row['available_modules'][j] for j in np.argsort(-score,kind='stable')[:5]]
 predictions[row['id']]=rec
validate_submission(selected,predictions,set(fm.module_ids))
with (ROOT/'audit/validation.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['id','recommendations']);w.writerows((s,' '.join(r)) for s,r in predictions.items())
with (ROOT/'audit/validation_solution.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['id','ground_truth']);w.writerows((r['id'],' '.join(f'{m}:{p}' for m,p in truth[r['id']].items())) for r in selected)
spec=importlib.util.spec_from_file_location('official',ROOT/'vendor/metric.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
original=json.loads((ROOT/'audit/original_predictions.json').read_text())
new=np.array([metric.pndcg_single(truth[r['id']],predictions[r['id']]) for r in selected]);old=np.array([metric.pndcg_single(truth[r['id']],original[r['id']]) for r in selected]);is_cold=np.array([not r['history'] for r in selected]);delta=new-old
rng=np.random.default_rng(991);ci=np.quantile([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)],[.025,.975]);ratio=sum(not r['history'] for r in te)/len(te)
result={'audit_n':len(selected),'old':float(old.mean()),'new':float(new.mean()),'difference':float(delta.mean()),'relative_improvement':float(new.mean()/old.mean()-1),'paired_bootstrap_95':ci.tolist(),'cold_n':int(is_cold.sum()),'new_cold':float(new[is_cold].mean()),'new_warm':float(new[~is_cold].mean()),'old_cold':float(old[is_cold].mean()),'old_warm':float(old[~is_cold].mean()),'test_cold_share':ratio,'new_reweighted_diagnostic':float(ratio*new[is_cold].mean()+(1-ratio)*new[~is_cold].mean()),'old_reweighted_diagnostic':float(ratio*old[is_cold].mean()+(1-ratio)*old[~is_cold].mean()),'reported_public_previous':0.30869875537635866,'public_new':None}
(ROOT/'audit/final_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
