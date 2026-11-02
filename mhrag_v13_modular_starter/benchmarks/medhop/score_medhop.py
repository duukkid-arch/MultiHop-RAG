#!/usr/bin/env python3
"""Offline evaluator: this script ONLY handles gold. Never import it into inference service.

Predictions JSONL: {"case_id":"medhop_dev_...","prediction":"drug name"}.
"""
import argparse,json,sys,unicodedata
from pathlib import Path

def lines(path):
    with path.open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{n}: {e}')

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC',str(s)).casefold().strip().split())

def read_unique(path,key):
    result={}
    for d in lines(path):
        if not isinstance(d,dict) or key not in d:raise ValueError(f'{path}: missing {key}')
        k=d[key]
        if k in result:raise ValueError(f'{path}: duplicate {key}={k}')
        result[k]=d
    return result

def score(gold_path,pred_path):
    gold=read_unique(gold_path,'case_id')
    preds=read_unique(pred_path,'case_id')
    if not gold:raise ValueError('empty gold')
    unknown=set(preds)-set(gold)
    if unknown:raise ValueError(f'unknown case IDs: {sorted(unknown)[:3]}')
    correct=0; missing=0
    for k,g in gold.items():
        if k not in preds:missing+=1;continue
        answer=preds[k].get('prediction')
        if answer is None:raise ValueError(f'{k}: missing prediction')
        correct+=norm(answer)==norm(g['answer'])
    return {'count':len(gold),'submitted':len(preds),'missing':missing,
            'correct':correct,'accuracy':round(correct/len(gold),6),
            'metric':'normalized exact match','note':'unsubmitted = wrong'}
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--gold',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    a=ap.parse_args()
    try: print(json.dumps(score(a.gold,a.predictions),indent=2,ensure_ascii=False))
    except (ValueError,OSError,KeyError) as e:sys.exit('REFUSED: '+str(e))
