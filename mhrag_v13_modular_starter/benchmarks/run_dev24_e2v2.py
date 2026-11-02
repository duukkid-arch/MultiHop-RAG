#!/usr/bin/env python3
"""24-case historical DEV only; four independently executed arms, no Gold.

NOT a sealed test or independent clinical adjudication. Source-bound results
are medical review *candidates*, not validated clinical answers.
"""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,os,random,socket,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.v13_ablations_runner import run_job as legacy_run_job
from benchmarks.v13_pilot_runner_v10 import read_existing,atomic_write_json,local_model_ok
ARMS=('F','E1','E2R','E3R')


def load_questions(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    if d.get('dataset_kind')!='historical_dev_24_unseen_by_DEV8_not_holdout':raise ValueError('not allowed DEV24 file')
    rows=d.get('cases')
    if not isinstance(rows,list) or len(rows)!=24 or len({r['case_id'] for r in rows})!=24:raise ValueError('expected 24 unique questions')
    for r in rows:
        if set(r)!={'case_id','question'} or not all(isinstance(v,str) for v in r.values()):raise ValueError('unexpected field (possible Gold)')
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(rows,seed=42):
    rng=random.Random(seed);out=[]
    for r in rows:
        perm=list(ARMS);rng.shuffle(perm)
        out.extend({'case_id':r['case_id'],'arm':a} for a in perm)
    return out


def make_manifest(args,plan):
    rel=['benchmarks/dev24_e2v2_NO_GOLD.json','benchmarks/run_dev24_e2v2.py',
         'mhrag_v13/evidence_to_answer_v2.py','mhrag_v13/ablation_arms.py',
         'mhrag_v13/grounded_controller.py','mhrag_v13/resolve.py',
         'mhrag_v13/grounded_answers.py','study_aliases.reviewed.json']
    return {'protocol':'v13-E2R-E3R-historical-DEV24-v1','dataset':'historical_DEV_NOT_holdout',
            'arms':list(ARMS),'plan':plan,
            'code_inputs_sha256':{p:digest(ROOT/p) for p in rel},
            'database_sha256':digest(args.db),
            'config':{'model':args.model,'seed':args.seed,'qwen_url':args.qwen_url,
                      'chat_url':args.chat_url,'max_hours':args.max_hours},
            'medical_gold':'NOT_LOADED','clinical_accuracy':'NOT_EVALUATED'}


def summarize(rows,plan):
    complete=(len(rows)==len(plan) and not any(r.get('status')=='error' for r in rows))
    arms={}
    for a in ARMS:
        rs=[r for r in rows if r['arm']==a];ok=[r for r in rs if r['status']=='ok']
        arms[a]={'completed':len(rs),'runtime_errors':len(rs)-len(ok),
                 'median_ms':round(statistics.median(r['wall_ms'] for r in ok),1) if ok else None,
                 'cases_with_candidates':None if a=='F' else sum((r.get('accepted_claim_count') or 0)>0 for r in ok),
                 'accepted_claims':None if a=='F' else sum(r.get('accepted_claim_count') or 0 for r in ok),
                 'decisions':dict(Counter((r.get('decision') or {}).get('decision') for r in ok)) if a!='F' else {},
                 'source_bound_cases':sum(r.get('source_bound_mode') is True for r in ok) if a in ('E2R','E3R') else None}
    return {'state':'complete' if complete else 'incomplete', 'planned':len(plan),'completed':len(rows),
            'errors':sum(r.get('status')=='error' for r in rows),'arms':arms,
            'clinical_accuracy':'NOT_EVALUATED','question_coverage_requires_human_PDF_review':True,
            'dataset':'HISTORICAL_DEV24_NO_GOLD_NOT_HOLDOUT'}


def export_reviews(out,rows,questions,seed):
    ordered=[];key=[];rng=random.Random(seed+101)
    for cid in questions:
        group=[r for r in rows if r['case_id']==cid];rng.shuffle(group)
        if len(group)!=4 or any(r['status']!='ok' for r in group):continue
        for i,r in enumerate(group):
            anon='ABCD'[i]
            ordered.append({'case_id':cid,'question':questions[cid],'blind_arm':anon,
                            'answer':r.get('answer',''),'source_ids':','.join(r.get('accepted_evidence_ids',[])),
                            'question_coverage_human':'','source_support_human':'','numeric_accuracy_human':'',
                            'appropriate_abstention_human':'','serious_error_human':'','reviewer_notes':''})
            key.append({'case_id':cid,'blind_arm':anon,'actual_arm':r['arm']})
    for name,records in [('review_R1.csv',ordered),('review_R2.csv',ordered),('ADMIN_unblind.csv',key)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,records[0].keys());w.writeheader();w.writerows(records)
    print('REVIEW_EXPORT_OK',len(ordered),'answers',flush=True)


def run_job(job,question,retriever,legacy,controllers):
    if job['arm'] in ('F','E1'):
        return legacy_run_job(job,question,retriever,legacy)
    start=time.monotonic()
    out={'case_id':job['case_id'],'arm':job['arm'],'status':'ok','utc':datetime.now(timezone.utc).isoformat()}
    try:
        result=controllers[job['arm']].run(question)
        if job['arm']=='E3R':
            from mhrag_v13.ablation_arms import e3_safety_review
            result=e3_safety_review(result,question)
        accepted=result['claims']['accepted']
        out.update({'answer':result['answer'],'result':result,'accepted_claim_count':len(accepted),
                    'accepted_evidence_ids':[x['claim']['evidence_id'] for x in accepted],
                    'decision':result['decision'],
                    'source_bound_mode':any(d.get('generation_mode')=='deterministic_source_bound' for d in result.get('generation_diagnostics',[])),
                    'generation_status':result.get('generation_status'),'generation_calls':result.get('generation_calls'),
                    'routing_mode':result.get('routing',{}).get('mode')})
    except (ValueError,TypeError,RuntimeError,OSError,KeyError,IndexError) as exc:
        out.update({'status':'error','error_type':type(exc).__name__,'error_message':str(exc)[:250]})
    out['wall_ms']=round((time.monotonic()-start)*1000,1)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='../mhrag_gpu_chat_musa/data/knowledge.sqlite')
    p.add_argument('--cases',default='benchmarks/dev24_e2v2_NO_GOLD.json')
    p.add_argument('--aliases',default='study_aliases.reviewed.json')
    p.add_argument('--out',default='logs/v13_e2v2_dev24_01')
    p.add_argument('--chat-url',default='http://127.0.0.1:7866')
    p.add_argument('--qwen-url',default='http://127.0.0.1:8002/v1')
    p.add_argument('--model',default='qwen-local')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-hours',type=float,default=2)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 0<a.max_hours<=2: p.error('max-hours requires (0,2]')
    for url in (a.chat_url,a.qwen_url):
        if not url.startswith(('http://127.0.0.1:','http://localhost:')):p.error('local endpoints only')
    for filename in (a.db,a.aliases,a.cases):
        if not Path(filename).is_file():p.error('missing '+filename)
    cases=load_questions(a.cases);questions={r['case_id']:r['question'] for r in cases}
    plan=make_plan(cases,a.seed)
    out=Path(a.out)
    if not a.execute:
        print(json.dumps({'preflight':'OK','cases':24,'tasks':96,'arms':ARMS,
                          'no_gold':True,'new_repair_arm':True,'clinical_accuracy':'NOT_EVALUATED'},ensure_ascii=False,indent=2));return
    from mhrag_v13.resolve import SQLiteStudyResolver
    from mhrag_v13.retrieve import LocalChatRetriever
    from mhrag_v13.grounded_answers import QwenJsonClient,ClaimSupportVerifier
    from mhrag_v13.grounded_controller import GroundedController
    from mhrag_v13.ablation_arms import QuestionAwareHarvester,AblationEngine
    from mhrag_v13.evidence_to_answer_v2 import SourceBoundFallbackGenerator
    local_model_ok(a.qwen_url,a.model)
    resolver=SQLiteStudyResolver(Path(a.db),a.aliases)
    retriever=LocalChatRetriever(a.chat_url)
    llm=QwenJsonClient(a.qwen_url,a.model)
    legacy=AblationEngine(resolver=resolver,retriever=retriever,db_path=a.db,llm=llm,max_calls=6,max_http_calls=4)
    controllers={k:GroundedController(resolver=resolver,retriever=retriever,db_path=a.db,
          harvester=QuestionAwareHarvester(a.db),generator=SourceBoundFallbackGenerator(llm,6),
          claim_verifier=ClaimSupportVerifier(),max_http_calls=4) for k in ('E2R','E3R')}
    frozen=make_manifest(a,plan)
    lock=ROOT.parent/'results'/'.v124_eval.lock';lock.parent.mkdir(exist_ok=True,parents=True)
    with lock.open('a+') as lf:
        try:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('ERROR: other GPU experiment active')
        if a.resume:
            if not (out/'manifest.json').is_file():raise SystemExit('no manifest to resume')
            if json.loads((out/'manifest.json').read_text(encoding='utf-8'))!=frozen:
                raise SystemExit('RESUME REFUSED: input/code/config mismatch')
        else:
            if out.exists():raise SystemExit('RESULT EXISTS: refuse overwrite')
            out.mkdir(parents=True);atomic_write_json(out/'manifest.json',frozen)
        rows=read_existing(out/'results.jsonl',plan)
        deadline=time.monotonic()+a.max_hours*3600
        with (out/'results.jsonl').open('a',encoding='utf-8') as f:
            for job in plan[len(rows):]:
                if time.monotonic()>=deadline:break
                row=run_job(job,questions[job['case_id']],retriever,legacy,controllers)
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                rows.append(row)
                print(f'[{len(rows)}/96] {job["case_id"]} {job["arm"]} {row["status"]} '+
                      f'accepted={row.get("accepted_claim_count")} {row["wall_ms"]}ms',flush=True)
                if len(rows)>=3 and all(r['status']=='error' for r in rows[-3:]):
                    print('STOP: 3 consecutive errors',flush=True);break
        report=summarize(rows,plan)
        if len(rows)<96:report['state']='time_cap_or_error_stop'
        atomic_write_json(out/'summary.json',report)
        if report['state']=='complete':export_reviews(out,rows,questions,a.seed)
        print('END '+json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
