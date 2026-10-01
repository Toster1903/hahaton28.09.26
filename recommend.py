#!/usr/bin/env python3
"""Reproducible priority-weighted course ranking with an audited LightGBM ensemble."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import (FeatureMaker,read_archive,choice_events,pndcg,write_submission,
                      CATS,NUMS,FEATURES,WEIGHTS)

CONTENT_FEATURES=['content_global_rate','content_recent_rate','content_institute_rate',
 'content_specialty_rate','smoothed_global_rate','smoothed_institute_rate',
 'smoothed_specialty_rate','course_age']
MODEL_CONFIGS=[{'name':'base','iterations':410,'weight':.25,'drop':CONTENT_FEATURES},
               {'name':'content','iterations':256,'weight':.75,'drop':[]}]


def cold_scores(maker,period):
    """Use the previous cohort; shrink a missing-profile cohort when it exists."""
    stats=maker.stats(period)
    counts,recent,_,_,ir,_,_,_,_,exposure,_,_,nrecent,ninst,_=stats
    prior=recent/max(nrecent,1)
    previous_available=np.array([period-1 in maker.modules[m]['available_in_periods']
                                 for m in maker.module_ids])
    # Older catalogs changed substantially; retain the validated most-recent-period prior.
    missing_count=ir['unknown']
    missing_exposure=previous_available*float(ninst['unknown'])
    return (missing_count+100*prior)/(missing_exposure+100)


def category_schema(train,val,test,modules):
    profiles=[r['profile'] for r in train+val+test]
    return {'module':sorted(m['module_id'] for m in modules),
            'institute':sorted({p.get('institute','') or 'unknown' for p in profiles}),
            'specialty':sorted({p.get('specialty','') or 'unknown' for p in profiles})}


def cast_categories(frame,schema):
    for col,values in schema.items():frame[col]=pd.Categorical(frame[col],categories=values)
    return frame


def fit_ensemble(maker,train,val,truth,model_dir,schema):
    period2=[(r,{c['module_id']:c['priority'] for h in r['history'] if h['period']==2
                for c in h['choices']}) for r in train if any(h['period']==2 for h in r['history'])]
    x2,y2,_=maker.frame(period2,2)
    x3,y3,_=maker.frame([(r,truth[r['id']]) for r in val],3)
    frame=cast_categories(pd.concat([x2,x3],ignore_index=True),schema)
    targets=np.r_[y2,y3];weights=np.r_[np.ones(len(y2)),np.full(len(y3),3.)]
    del x2,x3
    models=[]
    for config in MODEL_CONFIGS:
        model=lgb.LGBMRegressor(objective='cross_entropy',n_estimators=config['iterations'],
           learning_rate=.035,num_leaves=127,min_child_samples=60,colsample_bytree=.9,
           reg_lambda=8,verbosity=-1,random_state=42,n_jobs=6,metric='None')
        columns=[c for c in frame.columns if c not in config['drop']]
        print(f"Training {config['name']}: {len(frame)} pairs, {config['iterations']} trees",flush=True)
        model.fit(frame[columns],targets,sample_weight=weights)
        model.booster_.save_model(str(model_dir/f"{config['name']}.lgb"))
        models.append((model.booster_,config))
    return models


def predict(maker,records,models,schema,period=4):
    predictions={};fallback=cold_scores(maker,period)
    for offset in range(0,len(records),256):
        batch=records[offset:offset+256]
        frame,_,groups=maker.frame([(r,None) for r in batch],period)
        cast_categories(frame,schema)
        scores=np.zeros(len(frame),dtype=np.float64)
        for model,config in models:
            scores+=config['weight']*model.predict(frame[model.feature_name()])
        grouped={}
        for sid,module,score in zip(groups,frame['module'].astype(str),scores):
            grouped.setdefault(sid,[]).append((module,float(score)))
        for record in batch:
            sid=record['id']
            if not record['history'] and not any(record['profile'].values()):
                ranked=sorted(record['available_modules'],key=lambda m:(-fallback[maker.lookup[m]],m))[:5]
            else:
                ranked=[m for m,_ in sorted(grouped[sid],key=lambda item:(-item[1],item[0]))[:5]]
            predictions[sid]=ranked
        print(f"Predicted {min(offset+256,len(records))}/{len(records)} students",flush=True)
    return predictions


def validate_submission(records,predictions,known):
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(predictions):raise ValueError('Student IDs do not match')
    for row in records:
        recommendations=predictions[row['id']]
        if len(recommendations)!=5 or len(set(recommendations))!=5:
            raise ValueError(f"Need five distinct courses: {row['id']}")
        if not set(recommendations)<=set(row['available_modules'])&known:
            raise ValueError(f"Unavailable or unknown course for {row['id']}")


def main():
    root=Path(__file__).parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=root/'data/selection_formula.zip')
    parser.add_argument('--output',type=Path,default=root/'submission.csv')
    parser.add_argument('--model-dir',type=Path,default=root/'models')
    parser.add_argument('--predict-only',action='store_true')
    args=parser.parse_args()
    train,val,test,modules,truth=read_archive(args.archive)
    maker=FeatureMaker(modules,choice_events(train,val,truth))
    schema=category_schema(train,val,test,modules)
    args.model_dir.mkdir(parents=True,exist_ok=True)
    archive_hash=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if args.predict_only:
        manifest=json.loads((args.model_dir/'manifest.json').read_text())
        if manifest['archive_sha256']!=archive_hash:raise ValueError('Archive differs from the trained dataset')
        schema=manifest['categories']
        models=[(lgb.Booster(model_file=str(args.model_dir/f"{c['name']}.lgb")),c) for c in manifest['models']]
    else:
        models=fit_ensemble(maker,train,val,truth,args.model_dir,schema)
        manifest={'archive_sha256':archive_hash,'categories':schema,'models':MODEL_CONFIGS,
                  'seed':42,'period3_weight':3,'negative_sampling':False,
                  'lightgbm_version':lgb.__version__,'score_semantics':'expected priority-weighted relevance'}
        (args.model_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    predictions=predict(maker,test,models,schema)
    validate_submission(test,predictions,{m['module_id'] for m in modules})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_submission(args.output,test,predictions,modules)
    print(f"Verified submission: {args.output} ({len(test)} rows)",flush=True)

if __name__=='__main__':main()
