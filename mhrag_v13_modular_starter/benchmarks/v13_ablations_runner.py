#!/usr/bin/env python3
"""Independent historical DEV-only F/E1/E2/E3 randomized-per-case ablation.

Uses existing read-only SQLite, 7866 API and 8002 local Qwen. Never uses gold.
Each arm runs its own retrieval/generation, not a postprocessing relabel.
Shared lock with v12.4/v13 historical experiments; strict manifest + append-only resume.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.v13_pilot_runner_v10 import (load_cases,sha256_file,kb_info,local_model_ok,
                                                read_existing,atomic_write_json)

ARMS=('F','E1','E2','E3')


def make_plan(cases,seed):
    rng=random.Random(seed); plan=[]
    for c in cases:
        perm=list(ARMS);rng.shuffle(perm)
        plan.extend({'case_id':c['case_id'],'arm':a} for a in perm)
    return plan


def fingerprint_source(root):
    h=hashlib.sha256()
    paths=sorted((root/'mhrag_v13').glob('*.py')) + sorted((root/'benchmarks').glob('*.py'))
    for p in paths:
        h.update(p.relative_to(root).as_posix().encode());h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def manifest(args,cases,plan,info):
    return {
        'protocol':'v13-F-E1-E2-E3-DEV-ablation-prototype-v1',
        'kind':'historical_dev_91_not_holdout_NOT_MEDICALLY_VALIDATED',
        'arms':{
            'F':'existing conventional 7866 API, agentic=False',
            'E1':'study identity -> v13 original study-aware evidence -> generic quote/number-checked claims',
            'E2':'E1 identity/retrieval -> query-conditioned read-only evidence -> targeted quote/number-checked claims',
            'E3':'E2 -> additional task-coverage, study identity, universal and cross-paper safety gate',
        },
        'config':{'db':str(Path(args.db).resolve()),'cases':str(Path(args.cases).resolve()),
                  'aliases':str(Path(args.aliases).resolve()),'chat_url':args.chat_url,
                  'qwen_url':args.qwen_url,'model':args.model,'seed':args.seed,
                  'max_draft_calls':args.max_draft_calls,'max_http':args.max_http},
        'input_sha256':{
            'db':sha256_file(Path(args.db).resolve()),
            'cases':sha256_file(Path(args.cases).resolve()),
            'aliases':sha256_file(Path(args.aliases).resolve()),
            'source_tree':fingerprint_source(ROOT)},
        'kb':info,'plan':plan,'gold_loaded':False,'medical_gold_scoring':False,
    }


def run_job(job,question,retriever,engine):
    start=time.monotonic()
    out={'case_id':job['case_id'],'arm':job['arm'],'status':'ok',
         'utc':datetime.now(timezone.utc).isoformat()}
    try:
        if job['arm']=='F':
            result=retriever.search(question,agentic=False)
            out.update({'answer':result.answer,'api_doc_ids':sorted({s.document_id for s in result.sources}),
                        'http_calls':1,'accepted_claim_count':None,
                        'candidate_coverage':'not_measured_for_F'})
        else:
            result=engine.run(job['arm'],question)
            accepted=result.get('claims',{}).get('accepted',[])
            out.update({'answer':result['answer'],'http_calls':result.get('retrieval',{}).get('http_calls'),
                  'api_doc_ids':result['evidence']['api_doc_ids'],
                  'local_doc_ids':result['evidence']['local_doc_ids'],
                  'accepted_claim_count':len(accepted),
                  'accepted_study_ids':sorted({a['claim']['study_id'] for a in accepted}),
                  'generation_calls':result.get('generation_calls'),
                  'generation_status':result.get('generation_status'),
                  'routing_mode':result.get('routing',{}).get('mode'),
                  'decision':result.get('decision'),
                  'candidate_coverage':(result.get('ablation_safety') or {}).get('status',
                                       'not_measured_E1_E2'),
                  'result':result})
        out['medical_claims_verified']=False
    except Exception as exc:
        out.update({'status':'error','error_type':type(exc).__name__,
                    'error_message':str(exc)[:240]})
    out['wall_ms']=round((time.monotonic()-start)*1000,1)
    return out


def summarize(rows,plan,stopped_time=False,stopped_errors=False):
    stats={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        ok=[r for r in group if r['status']=='ok']
        stat={'completed':len(group),'errors':len(group)-len(ok),
              'median_ms':round(statistics.median(r['wall_ms'] for r in ok),1) if ok else None}
        if arm!='F':
            stat.update({'total_accepted_claims':sum(r.get('accepted_claim_count') or 0 for r in ok),
               'cases_with_accepted_claim':sum((r.get('accepted_claim_count') or 0)>0 for r in ok),
               'routing_modes':dict(Counter(r.get('routing_mode') for r in ok)),
               'decisions':dict(Counter((r.get('decision') or {}).get('decision') for r in ok)),
               'coverage_statuses':dict(Counter(r.get('candidate_coverage') for r in ok)),
               'generation_statuses':dict(Counter(r.get('generation_status') for r in ok))})
        stats[arm]=stat
    keyed={(r['case_id'],r['arm']) for r in rows if r['status']=='ok'}
    cases={p['case_id'] for p in plan}
    complete_pairs=sum(all((cid,arm) in keyed for arm in ARMS) for cid in cases)
    state=('stopped_errors' if stopped_errors else 'time_cap' if stopped_time else
           'complete' if len(rows)==len(plan) else 'incomplete')
    return {'state':state,'planned':len(plan),'completed':len(rows),
            'all_four_arms_successful_cases':complete_pairs,
            'total_runtime_errors':sum(r['status']=='error' for r in rows),
            'arms':stats,'medical_accuracy':'NOT_EVALUATED',
            'provisional_scope':'historical DEV diagnostics, NOT sealed holdout or formal trial',
            'cannot_claim_E3_better_without_blinded_adjudication':True}


def main():
    p=argparse.ArgumentParser(description='Read-only DEV 91 independent F/E1/E2/E3 study-aware ablation')
    p.add_argument('--db',required=True)
    p.add_argument('--cases',required=True)
    p.add_argument('--aliases',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--chat-url',default='http://127.0.0.1:7866')
    p.add_argument('--qwen-url',default='http://127.0.0.1:8002/v1')
    p.add_argument('--model',default='qwen-local')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-draft-calls',type=int,default=6)
    p.add_argument('--max-http',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=2)
    p.add_argument('--max-consecutive-errors',type=int,default=3)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if not 0<args.max_hours<=3:p.error('max-hours must be (0,3]; resume for longer runs')
    if not 1<=args.max_draft_calls<=10 or not 1<=args.max_http<=6:p.error('invalid call budget')
    if not 1<=args.max_consecutive_errors<=5:p.error('max-consecutive-errors 1..5')
    if args.resume and not args.execute:p.error('resume requires execute')
    if not all(Path(x).is_file() for x in (args.db,args.cases,args.aliases)):
        p.error('missing DB / DEV dataset / aliases')
    cases=load_cases(Path(args.cases))
    if len(cases)!=91 or json.loads(Path(args.cases).read_text(encoding='utf-8')).get('dataset_kind')!='historical_dev_91_no_gold':
        p.error('This runner is restricted to the 91-case historical DEV dataset')
    plan=make_plan(cases,args.seed)
    info=kb_info(Path(args.db).resolve())
    if not args.execute:
        print(json.dumps({'dry_run':True,'cases':len(cases),'jobs':len(plan),
            'arms':ARMS,'first_jobs':plan[:8],'gold_loaded':False,
            'kb':info,'warning':'Not a 2417-job formal test; no clinical scoring'},ensure_ascii=False,indent=2))
        return
    from mhrag_v13.resolve import SQLiteStudyResolver
    from mhrag_v13.retrieve import LocalChatRetriever
    from mhrag_v13.grounded_answers import QwenJsonClient
    from mhrag_v13.ablation_arms import AblationEngine
    local_model_ok(args.qwen_url,args.model)
    retriever=LocalChatRetriever(args.chat_url)
    engine=AblationEngine(resolver=SQLiteStudyResolver(Path(args.db),args.aliases),
        retriever=retriever,db_path=args.db,
        llm=QwenJsonClient(args.qwen_url,args.model),max_calls=args.max_draft_calls,
        max_http_calls=args.max_http)
    frozen=manifest(args,cases,plan,info)
    output=Path(args.out).resolve()
    lock_path=ROOT.parent/'results'/'.v124_eval.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+',encoding='utf-8') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another experiment holds the shared lock; do not start concurrently')
        if args.resume:
            if not (output/'manifest.json').is_file():raise SystemExit('RESUME REFUSED: missing manifest')
            old=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            if any(old.get(k)!=frozen[k] for k in frozen):
                raise SystemExit('RESUME REFUSED: frozen dataset/code/config/DB fingerprints differ')
        else:
            if output.exists():raise SystemExit('Refusing to overwrite previous experiment output')
            output.mkdir(parents=True)
            atomic_write_json(output/'manifest.json', {**frozen,
                'created_utc':datetime.now(timezone.utc).isoformat()})
        rows=read_existing(output/'results.jsonl',plan)
        consecutive=0
        for prev in reversed(rows):
            if prev['status']!='error':break
            consecutive+=1
        if consecutive>=args.max_consecutive_errors:
            raise SystemExit('Too many preceding runtime errors; inspect results before resuming')
        lookup={c['case_id']:c['question'] for c in cases}
        deadline=time.monotonic()+args.max_hours*3600
        stopped_time=False;stopped_errors=False
        with (output/'results.jsonl').open('a',encoding='utf-8') as f:
            for i in range(len(rows),len(plan)):
                if time.monotonic()>=deadline:stopped_time=True;break
                job=plan[i]
                row=run_job(job,lookup[job['case_id']],retriever,engine)
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                rows.append(row)
                consecutive=consecutive+1 if row['status']=='error' else 0
                print(f'[{len(rows)}/{len(plan)}] {row["case_id"]} {row["arm"]} {row["status"]} '
                      f'{row["wall_ms"]:.0f}ms accepted={row.get("accepted_claim_count")}',flush=True)
                if consecutive>=args.max_consecutive_errors:stopped_errors=True;break
        result=summarize(rows,plan,stopped_time,stopped_errors)
        atomic_write_json(output/'summary.json',result)
        print('END '+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
