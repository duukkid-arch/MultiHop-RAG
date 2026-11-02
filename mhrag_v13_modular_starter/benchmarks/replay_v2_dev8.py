#!/usr/bin/env python3
"""Offline DEV8 evidence-to-answer replay; immutable previously uploaded spans."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mhrag_v13.evidence_slots import EvidenceSpan
from mhrag_v13.evidence_to_answer_v2 import source_bound_generate

def replay(source_dir,out):
    src=Path(source_dir);out=Path(out)
    cases=sorted(src.glob('*.json'))
    if len(cases)!=8:raise ValueError('expected 8 frozen historical DEV source snapshots')
    if out.exists():raise FileExistsError('refusing to overwrite '+str(out))
    out.mkdir(parents=True)
    rows=[]
    for p in cases:
        packet=json.loads(p.read_text(encoding='utf-8'))
        if packet.get('no_gold') is not True:raise ValueError('Gold data potentially present')
        spans=[EvidenceSpan(**s) for s in packet['spans']]
        result=source_bound_generate(packet['question'],spans)
        row={'case_id':packet['case_id'],'question':packet['question'],'status':result['status'],
             'candidate':result['claims'][0]['text'] if result['claims'] else '',
             'evidence_id':result.get('evidence_id',''),'medical_accuracy_human':'','source_support_human':''}
        rows.append(row)
    with (out/'dev8_candidates_for_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    summary={'dataset':'historical_DEV8_reused_for_design','cases':len(rows),
            'source_bound_candidates':sum(bool(r['candidate']) for r in rows),
            'medical_accuracy':'NOT_EVALUATED','note':'Pattern-debugged DEV8. Do not use as unbiased performance estimate.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sources',default='logs/v13_e2_quote_probe8_01/frozen_sources')
    p.add_argument('--out',default='logs/v13_e2v2_replay8_01')
    a=p.parse_args();replay(a.sources,a.out)
