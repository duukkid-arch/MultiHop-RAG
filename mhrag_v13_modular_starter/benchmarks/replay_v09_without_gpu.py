#!/usr/bin/env python3
"""Re-score existing v0.9 accepted claims with v1.0 coverage, no live calls."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mhrag_v13.question_coverage_v10 import evaluate

def replay(cases,results):
    questions={c['case_id']:c['question'] for c in cases['cases']}
    rows=[]
    for r in results:
        if r.get('arm')!='G':continue
        q=questions[r['case_id']]
        result=r.get('result') or {}
        checked=(result.get('claims') or {}).get('accepted',[])
        new,aligned=evaluate(q,checked,result.get('identity'),result.get('routing'))
        rows.append({'case_id':r['case_id'], 'old_status':r.get('task_coverage_status'),
                     'new_status':new['status'], 'question_kind_v10':new['kind'],
                     'new_reasons':';'.join(new['reasons']),
                     'prechecked_claims':len(checked),'aligned_claims_v10':len(aligned),
                     'human_medical_accuracy':'NOT_EVALUATED'})
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',required=True)
    ap.add_argument('--results',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    cases=json.loads(Path(args.questions).read_text(encoding='utf-8'))
    data=[json.loads(x) for x in Path(args.results).read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=replay(cases,data)
    if len(rows)!=16:raise SystemExit('EXPECTED exactly 16 existing historical G results')
    out=Path(args.out)
    if out.exists():raise SystemExit('Refuse overwrite '+str(out))
    out.mkdir(parents=True)
    with (out/'replay.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'kind':'offline_dev_replay_no_GPU_no_medical_truth','cases':len(rows),
            'old_statuses':dict(Counter(x['old_status'] for x in rows)),
            'new_statuses':dict(Counter(x['new_status'] for x in rows)),
            'downgraded':[(x['case_id'],x['old_status'],x['new_status']) for x in rows if x['old_status']=='candidate_task_covered_NOT_MEDICALLY_VERIFIED' and x['new_status']!='candidate_task_covered_NOT_MEDICALLY_VERIFIED'],
            'medical_accuracy':'NOT_EVALUATED',
            'warning':'Stricter coverage is not evidence of medical performance improvement.'}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
