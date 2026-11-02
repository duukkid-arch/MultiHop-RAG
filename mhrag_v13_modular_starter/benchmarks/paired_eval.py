#!/usr/bin/env python3
"""Small DEV paired F vs modular-v13 evaluation; no training or gold leakage.

Requires an existing v12.4 HTTP API and frozen SQLite KB. Uses the same
benchmark lock as the v12.5-C runner to prevent concurrent GPU timings.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import random
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mhrag_v13.resolve import SQLiteStudyResolver
from mhrag_v13.retrieve import LocalChatRetriever
from mhrag_v13.verify import StructuralEvidenceGate
from mhrag_v13.answer import EvidencePacketAnswerer
from mhrag_v13.controller import AdaptiveController
from mhrag_v13.tracing import JsonlTrace


def load_cases(path, phases):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    data=doc['cases'] if isinstance(doc,dict) else doc
    rows=[c for c in data if c.get('phase') in phases]
    if len({c['case_id'] for c in rows})!=len(rows):raise ValueError('Duplicate case ID')
    for c in rows:
        if not isinstance(c.get('question'),str) or not isinstance(c.get('gold_ids'),list):
            raise ValueError('Required keys: case_id, question, gold_ids, phase')
    return rows


def mean(values):return round(statistics.mean(values),5) if values else None


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--db',required=True)
    a.add_argument('--cases',required=True)
    a.add_argument('--out',required=True)
    a.add_argument('--phases',default='baseline')
    a.add_argument('--limit',type=int,default=16)
    a.add_argument('--max-hours',type=float,default=3.5)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--dry-run',action='store_true')
    a.add_argument('--chat-url',default='http://127.0.0.1:7866')
    args=a.parse_args()
    cases=load_cases(args.cases,set(args.phases.split(',')))
    random.Random(args.seed).shuffle(cases)
    if args.limit>0:cases=cases[:args.limit]
    if args.dry_run:
        print(json.dumps({'cases':len(cases),'planned_jobs':len(cases)*2,
              'note':'F and C; C may make up to 4 internal HTTP requests; DEV only'},ensure_ascii=False,indent=2))
        return
    if args.max_hours<=0 or args.max_hours>4:raise ValueError('Max hours must be (0,4]')
    out=Path(args.out)
    if out.exists():raise FileExistsError(f'Refuse to overwrite existing results: {out}')
    out.mkdir(parents=True)
    # Same lock used by the user's other experimental runs; never steal it.
    lock_path=out.parent/'.v124_eval.lock'
    lock=lock_path.open('a')
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another evaluation owns the lock; do not run concurrent experiments')
        retriever=LocalChatRetriever(args.chat_url)
        c=AdaptiveController(SQLiteStudyResolver(args.db),retriever,StructuralEvidenceGate(),
             EvidencePacketAnswerer(), trace=JsonlTrace(out/'events.jsonl'),max_http_calls=4)
        deadline=time.monotonic()+args.max_hours*3600
        rows=[];state='completed'
        with (out/'results.jsonl').open('w',encoding='utf-8') as f:
            for case in cases:
                for arm in random.Random(args.seed+len(rows)).sample(['F','C'],2):
                    if time.monotonic()>=deadline:
                        state='time_cap';break
                    start=time.monotonic();error=None;ids=[];calls=1;status='ok'
                    try:
                        if arm=='F':
                            data=retriever.search(case['question'])
                            ids=sorted({s.document_id for s in data.sources})
                        else:
                            data=c.run(case['question'])
                            ids=data['observed_ids'];calls=data['http_calls'];status=data['assessment']
                    except Exception as exc:
                        error=type(exc).__name__;status='error'
                    gold=set(case['gold_ids'])  # GOLD ONLY ENTERS EVALUATION, never resolver/retriever/controller.
                    row={'case_id':case['case_id'],'phase':case['phase'],'arm':arm,
                          'observed_ids':ids,'coverage':len(gold&set(ids))/len(gold) if gold else None,
                          'all_gold':gold.issubset(ids) if gold else None,'status':status,
                          'error_type':error,'http_calls':calls,
                          'wall_ms':round((time.monotonic()-start)*1000,1)}
                    rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                    print(f"[{len(rows)}/{len(cases)*2}] {case['case_id']} {arm} gold={row['coverage']} {row['wall_ms']:.0f}ms",flush=True)
                if state=='time_cap':break
        groups={}
        for arm in ['F','C']:
            rs=[r for r in rows if r['arm']==arm and r['error_type'] is None]
            groups[arm]={'n':len(rs),'mean_gold_coverage':mean([r['coverage'] for r in rs if r['coverage'] is not None]),
                         'median_wall_ms':statistics.median([r['wall_ms'] for r in rs]) if rs else None,
                         'total_api_calls':sum(r['http_calls'] for r in rs)}
        keyed={(r['case_id'],r['arm']):r for r in rows if r['error_type'] is None}
        paired=[(keyed[(c['case_id'],'C')]['coverage']-keyed[(c['case_id'],'F')]['coverage']) for c in cases
               if (c['case_id'],'F') in keyed and (c['case_id'],'C') in keyed]
        summary={'state':state,'planned':len(cases)*2,'completed':len(rows),
                 'errors':sum(bool(r['error_type']) for r in rows),'groups':groups,
                 'paired_questions':len(paired),'mean_C_minus_F':mean(paired),
                 'limitation':'DEV exploration: coverage != factual accuracy; no statistical claim from short pilot'}
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:lock.close()

if __name__=='__main__':main()
