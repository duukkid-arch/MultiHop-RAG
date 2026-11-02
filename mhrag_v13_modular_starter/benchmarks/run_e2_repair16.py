#!/usr/bin/env python3
"""DEV-only real paired E2(old) vs E2(repaired). E3 repair is SAME-RESULT offline gate."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,os,random,secrets,statistics,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.v13_pilot_runner_v10 import sha256_file,kb_info,local_model_ok,read_existing,atomic_write_json
from benchmarks.v13_ablations_runner import fingerprint_source

ARMS=('E2_legacy','E2_repaired')


def load_cases(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('dataset_kind')!='historical_dev_16_regression_no_gold':raise ValueError('DEV-only data required')
    cases=obj.get('cases',[])
    if len(cases)!=16 or len({c['case_id'] for c in cases})!=16:raise ValueError('Expect 16 unique dev cases')
    for c in cases:
        if not isinstance(c.get('question'),str) or not c['question'].strip():raise ValueError('invalid question')
        if set(c)-{'case_id','question','case_type'}:raise ValueError('Gold/answer keys not allowed in data')
    return cases


def plan_for(cases,seed):
    rng=random.Random(seed)
    out=[]
    for c in cases:
        arms=list(ARMS);rng.shuffle(arms)
        out.extend({'case_id':c['case_id'],'arm':a} for a in arms)
    return out


def make_summary(rows,plan,*,time_cap=False,failed=False):
    arms={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];ok=[r for r in rr if r['status']=='ok']
        reasons=Counter();coverage=Counter();e3=Counter()
        for r in ok:
            for d in r.get('generation_diagnostics',[]):reasons[d.get('status','unknown')]+=1
            if arm=='E2_repaired':
                coverage[r.get('e3_posthoc_coverage','unknown')]+=1
                e3[r.get('e3_posthoc_decision','unknown')]+=1
        arms[arm]={'completed':len(rr),'errors':len(rr)-len(ok),
            'median_ms':round(statistics.median(r['wall_ms'] for r in ok),1) if ok else None,
            'cases_with_accepted_claim':sum((r.get('accepted_claim_count') or 0)>0 for r in ok),
            'total_accepted_claims':sum(r.get('accepted_claim_count') or 0 for r in ok),
            'generation_statuses':dict(Counter(r.get('generation_status') for r in ok)),
            'diagnostic_events':dict(reasons),
            'e3_posthoc_coverage_NOT_MEDICAL_TRUTH':dict(coverage) if arm=='E2_repaired' else None,
            'e3_posthoc_decisions':dict(e3) if arm=='E2_repaired' else None}
    success={(r['case_id'],r['arm']) for r in rows if r['status']=='ok'}
    return {'state':'stopped_errors' if failed else 'time_cap' if time_cap else 'complete' if len(rows)==len(plan) else 'incomplete',
        'planned':len(plan),'completed':len(rows),'paired_cases_completed':sum(all((p,a) in success for a in ARMS) for p in {x['case_id'] for x in plan}),
        'total_runtime_errors':sum(r['status']=='error' for r in rows),'arms':arms,
        'medical_accuracy':'NOT_EVALUATED','task_answered_correctly':'NOT_EVALUATED',
        'e3_posthoc':'Applied to repaired E2 results without extra retrieval or LLM calls; NOT an independent generation arm',
        'dataset':'selected historical DEV cases, not sealed holdout',
        'warning':'Accepted claims and coverage are automated prechecks; expert PDF adjudication needed'}


def execute_job(job,question,legacy,repaired):
    start=time.monotonic();out={'case_id':job['case_id'],'arm':job['arm'],'status':'ok',
                                'utc':datetime.now(timezone.utc).isoformat()}
    try:
        if job['arm']=='E2_legacy':result=legacy.run('E2',question)
        else:result=repaired.run(question)
        accepted=(result.get('claims') or {}).get('accepted',[])
        out.update({'answer':result.get('answer',''),
            'accepted_claim_count':len(accepted),'generation_calls':result.get('generation_calls'),
            'generation_status':result.get('generation_status'),
            'generation_diagnostics':result.get('generation_diagnostics') or [],
            'routing_mode':(result.get('routing') or {}).get('mode'),
            'decision':result.get('decision'),
            'api_doc_ids':(result.get('evidence') or {}).get('api_doc_ids',[]),
            'local_doc_ids':(result.get('evidence') or {}).get('local_doc_ids',[]),
            'result':result})
        if job['arm']=='E2_repaired':
            from mhrag_v13.e2_repair_trial import posthoc_e3_from_repaired
            e3=posthoc_e3_from_repaired(result,question)
            out['e3_posthoc_answer']=e3['answer']
            out['e3_posthoc_decision']=e3['decision']['decision']
            out['e3_posthoc_coverage']=e3['ablation_safety']['status']
    except Exception as exc:
        out.update({'status':'error','error_type':type(exc).__name__,
                    'error_message':str(exc)[:200]})
    out['wall_ms']=round((time.monotonic()-start)*1000,1)
    return out


def export_review(outdir):
    outdir=Path(outdir);summary=json.loads((outdir/'summary.json').read_text(encoding='utf-8'))
    if summary['state']!='complete' or summary['total_runtime_errors']!=0:raise ValueError('Not complete/error-free')
    dest=outdir.parent/(outdir.name+'_review')
    if dest.exists():raise ValueError('Review output exists; will not overwrite')
    mani=json.loads((outdir/'manifest.json').read_text(encoding='utf-8'))
    questions={c['case_id']:c['question'] for c in load_cases(mani['config']['cases'])}
    rows=[json.loads(s) for s in (outdir/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows)!=32 or len({(r['case_id'],r['arm']) for r in rows})!=32:raise ValueError('Bad result count')
    keyed={(r['case_id'],r['arm']):r for r in rows}
    rng=random.Random(secrets.randbits(64))
    blind=[];mapping=[]
    for cid,q in questions.items():
        variants=[('E2_legacy',keyed[(cid,'E2_legacy')]['answer']),
                  ('E2_repaired',keyed[(cid,'E2_repaired')]['answer'])]
        rng.shuffle(variants)
        for i,(a,answer) in enumerate(variants,1):
            name=cid+':A'+str(i)
            blind.append({'case_id':cid,'blind_id':name,'question':q,'answer':answer,
                'directly_answers_0_1_2':'','medical_fact_check_from_PDF':'',
                'quote_supported_yes_no_uncertain':'','abstention_appropriate_yes_no_na':'',
                'verified_source_pages':'','notes':''})
            mapping.append({'case_id':cid,'blind_id':name,'arm':a})
    dest.mkdir(parents=True)
    for reviewer in ('R1','R2'):
        with (dest/f'blind_{reviewer}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(blind[0]));w.writeheader();w.writerows(blind)
    with (dest/'ADMIN_MAPPING_ONLY_AFTER_REVIEW.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    (dest/'README.txt').write_text('历史DEV 16题、双臂32份答案。医学事实需独立审阅原PDF。部分答案格式可能暴露算法身份；审阅前可统一展示。E3 posthoc仅内部诊断，不作为第三组盲审。\n',encoding='utf-8')
    print('REVIEW_EXPORT_OK',str(dest),'rows=',len(blind))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--aliases',required=True);p.add_argument('--out',required=True)
    p.add_argument('--chat-url',default='http://127.0.0.1:7866');p.add_argument('--qwen-url',default='http://127.0.0.1:8002/v1')
    p.add_argument('--model',default='qwen-local');p.add_argument('--max-calls',type=int,default=6)
    p.add_argument('--max-hours',type=float,default=1);p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--export-review',action='store_true')
    a=p.parse_args()
    if a.export_review:
        export_review(Path(a.out));return
    if a.resume and not a.execute:p.error('resume requires execute')
    if not 0<a.max_hours<=2 or not 1<=a.max_calls<=10:p.error('invalid budget')
    if not all(Path(s).is_file() for s in (a.db,a.cases,a.aliases)):p.error('missing sources')
    cases=load_cases(a.cases);plan=plan_for(cases,a.seed)
    if not a.execute:
        print(json.dumps({'dry_run':True,'cases':len(cases),'jobs':len(plan),'arms':list(ARMS),'seed':a.seed,
             'e3_is_posthoc':True,'gold_loaded':False,'medical_gold_scoring':False},ensure_ascii=False,indent=2));return
    local_model_ok(a.qwen_url,a.model)
    from mhrag_v13.resolve import SQLiteStudyResolver
    from mhrag_v13.retrieve import LocalChatRetriever
    from mhrag_v13.grounded_answers import QwenJsonClient
    from mhrag_v13.ablation_arms import AblationEngine
    from mhrag_v13.e2_repair_trial import RepairEngine
    from benchmarks.v13_ablations_runner import fingerprint_source
    retriever=LocalChatRetriever(a.chat_url)
    resolver=SQLiteStudyResolver(Path(a.db),a.aliases)
    model=QwenJsonClient(a.qwen_url,a.model)
    legacy=AblationEngine(resolver=resolver,retriever=retriever,db_path=a.db,llm=model,max_calls=a.max_calls)
    repaired=RepairEngine(resolver=resolver,retriever=retriever,db_path=a.db,llm=model,max_calls=a.max_calls)
    out=Path(a.out).resolve()
    manifest={'protocol':'historical_DEV_E2_repair_paired_v1','kind':'historical_dev_16_no_gold',
              'config':{'db':str(Path(a.db).resolve()),'cases':str(Path(a.cases).resolve()),
                  'aliases':str(Path(a.aliases).resolve()),'qwen_url':a.qwen_url,'chat_url':a.chat_url,
                  'model':a.model,'max_calls':a.max_calls,'seed':a.seed},
              'hashes':{'db':sha256_file(Path(a.db).resolve()),'cases':sha256_file(Path(a.cases).resolve()),
                  'aliases':sha256_file(Path(a.aliases).resolve()),'source':fingerprint_source(ROOT)},
              'plan':plan,'medical_gold_scoring':False,'e3_posthoc':True}
    lock=ROOT.parent/'results'/'.v124_eval.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Evaluation lock held: do not run concurrently')
        if a.resume:
            old=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            if any(old.get(k)!=manifest.get(k) for k in manifest):raise SystemExit('RESUME REFUSED: changed data/code/config')
        else:
            if out.exists():raise SystemExit('Output exists; refusing to overwrite; use resume if interrupted')
            out.mkdir(parents=True)
            atomic_write_json(out/'manifest.json',{**manifest,'created_utc':datetime.now(timezone.utc).isoformat()})
        rows=read_existing(out/'results.jsonl',plan)
        bad=0
        for r in reversed(rows):
            if r['status']!='error':break
            bad+=1
        if bad>=3:raise SystemExit('3 consecutive runtime errors: inspect before resuming')
        deadline=time.monotonic()+a.max_hours*3600;stopped_time=False;stopped_errors=False
        question_map={c['case_id']:c['question'] for c in cases}
        with (out/'results.jsonl').open('a',encoding='utf-8') as f:
            for i in range(len(rows),len(plan)):
                if time.monotonic()>=deadline:stopped_time=True;break
                pjob=plan[i]
                result=execute_job(pjob,question_map[pjob['case_id']],legacy,repaired)
                f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                rows.append(result);bad=bad+1 if result['status']=='error' else 0
                print(f'[{len(rows)}/32] {pjob["case_id"]} {pjob["arm"]} {result["status"]} '
                      f'{result["wall_ms"]:.0f}ms accepted={result.get("accepted_claim_count")}',flush=True)
                if bad>=3:stopped_errors=True;break
        summ=make_summary(rows,plan,time_cap=stopped_time,failed=stopped_errors)
        atomic_write_json(out/'summary.json',summ)
        print('END '+json.dumps(summ,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
