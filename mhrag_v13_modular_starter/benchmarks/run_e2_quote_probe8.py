#!/usr/bin/env python3
"""Fixed-source paired DEV diagnostic, not medical benchmarking or independent holdout.

A: existing RepairGenerator direct Chinese answer.
B: copy an exact English source quote (evidence extraction, NOT answer generation).
Same studies, ranked snippets, max model calls. Sources are captured ONCE per case
and frozen so both arms get identical evidence, even across resumptions.
"""
from __future__ import annotations
import argparse,dataclasses,fcntl,hashlib,json,os,random,statistics,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def hash_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_cases(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    if d.get('dataset_kind')!='historical_dev_8_prompt_diagnostic_no_gold':raise ValueError('dev-only cases required')
    cases=d.get('cases',[])
    if len(cases)!=8 or len({c['case_id'] for c in cases})!=8:raise ValueError('expected 8 unique cases')
    for case in cases:
        if set(case)-{'case_id','question','case_type'}:raise ValueError('gold/answer/doc label in cases')
        if not case['case_id'].replace('_','').isalnum():raise ValueError('invalid ID')
    return cases


def immutable_config(a):
    paths=[Path(a.db).resolve(),Path(a.cases).resolve(),Path(a.aliases).resolve(),
           ROOT/'mhrag_v13/e2_quote_probe.py',ROOT/'mhrag_v13/e2_repair_trial.py',
           ROOT/'benchmarks/run_e2_quote_probe8.py']
    return {'paths':{str(p):hash_file(p) for p in paths},'model':a.model,
            'chat_url':a.chat_url,'qwen_url':a.qwen_url,'max_calls':a.max_calls,'seed':a.seed}


def write_atomic(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def source_for_case(question,case_id,folder,resolver,controller,harvester,db):
    from mhrag_v13.routing import decide_route
    from mhrag_v13.evidence_slots import EvidenceSpan,Slot
    path=folder/(case_id+'.json')
    if path.exists():
        packet=json.loads(path.read_text(encoding='utf-8'))
        if packet['question']!=question:raise RuntimeError('frozen source question differs')
        return packet,[EvidenceSpan(**x) for x in packet['spans']],[Slot(**x) for x in packet['slots']]
    # Retrieve exactly once per question, then both arms use frozen same original source spans.
    retrieval=controller.retrieval.run(question)
    identity=resolver.resolve(question)
    route=decide_route(identity,retrieval['sources'],db)
    spans,slots=harvester.harvest(route.document_ids,question)
    packet={'case_id':case_id,'question':question,'route':route.to_dict(),
            'identity':{'required_ids':list(identity.ids),'unresolved':list(identity.unresolved)},
            'retrieval_http_calls':retrieval.get('http_calls'),
            'spans':[s.to_dict() for s in spans],
            'slots':[s.to_dict() for s in slots],
            'no_gold':True}
    write_atomic(path,packet)
    return packet,spans,slots


def summarize(rows,expected=16):
    arms={}
    for arm in ('direct','verbatim'):
        rr=[r for r in rows if r['arm']==arm]
        ok=[r for r in rr if r['status']=='ok']
        events=Counter()
        for r in ok:
            for d in r.get('diagnostics',[]):events[d['status']]+=1
        key='accepted_count' if arm=='direct' else 'accepted_quote_count'
        arms[arm]={'completed':len(rr),'errors':len(rr)-len(ok),
                   'cases_with_candidate':sum((r.get(key) or 0)>0 for r in ok),
                   'total_candidates':sum(r.get(key) or 0 for r in ok),
                   'llm_calls':sum(r.get('generation_calls') or 0 for r in ok),
                   'median_ms':round(statistics.median(r['wall_ms'] for r in ok),1) if ok else None,
                   'diagnostic_events':dict(events)}
    done=len(rows)==expected and all(x['status']=='ok' for x in rows)
    return {'state':'complete' if done else 'incomplete','planned':expected,'completed':len(rows),
      'paired_cases':sum({('direct',cid),('verbatim',cid)}.issubset({(r['arm'],r['case_id']) for r in rows if r['status']=='ok'}) for cid in {r['case_id'] for r in rows}),
      'arms':arms,'medical_accuracy':'NOT_EVALUATED','question_answer_correctness':'NOT_EVALUATED',
      'warning':'Verbatim arm extracts candidates, NOT final answer. Do not compare candidate counts as medical accuracy or answer quality.',
      'design':'same frozen SQLite source spans per case; distinct prompts; same 4-LLM-call budget'}


def run(a):
    from mhrag_v13.resolve import SQLiteStudyResolver
    from mhrag_v13.retrieve import LocalChatRetriever
    from mhrag_v13.grounded_controller import GroundedController
    from mhrag_v13.grounded_answers import QwenJsonClient
    from mhrag_v13.e2_repair_trial import RepairHarvester
    from mhrag_v13.e2_quote_probe import QuoteProbe,baseline
    cases=load_cases(a.cases);out=Path(a.out).resolve();manifest=out/'manifest.json'
    config=immutable_config(a)
    if a.resume:
        if not manifest.is_file():raise RuntimeError('cannot resume without manifest')
        prev=json.loads(manifest.read_text(encoding='utf-8'))
        if prev['config']!=config:raise RuntimeError('SOURCE/DB/CONFIG FINGERPRINT CHANGED: resume refused')
    else:
        if out.exists():raise RuntimeError('Existing result directory, refusing to overwrite')
        out.mkdir(parents=True);write_atomic(manifest,{'protocol':'E2_SOURCE_QUOTE_DIAGNOSTIC_8_DEV',
                'config':config,'no_gold':True,'case_ids':[c['case_id'] for c in cases]})
    lock=ROOT.parent/'results'/'.v124_eval.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as lk:
        try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another evaluation is active; refuse GPU overlap')
        # No Qwen on dry-run; this happens only after --execute.
        import urllib.request
        req=urllib.request.Request(a.qwen_url.rstrip('/')+'/models')
        with urllib.request.urlopen(req,timeout=12) as rsp:
            if rsp.status!=200:raise RuntimeError('model endpoint unreachable')
        resolver=SQLiteStudyResolver(Path(a.db),a.aliases)
        retriever=LocalChatRetriever(a.chat_url)
        ctrl=GroundedController(resolver=resolver,retriever=retriever,db_path=a.db)
        harvester=RepairHarvester(a.db)
        llm=QwenJsonClient(a.qwen_url,a.model)
        quote_probe=QuoteProbe(llm,max_calls=a.max_calls)
        rows=[];resfile=out/'results.jsonl'
        if resfile.exists():
            rows=[json.loads(s) for s in resfile.read_text(encoding='utf-8').splitlines() if s.strip()]
        seen={(r['case_id'],r['arm']) for r in rows}
        if len(rows)!=len(seen):raise RuntimeError('duplicate results; refusing resume')
        (out/'frozen_sources').mkdir(exist_ok=True)
        began=time.monotonic()
        for case in cases:
            cid=case['case_id'];remaining=[arm for arm in ('direct','verbatim') if (cid,arm) not in seen]
            if not remaining:continue
            if (time.monotonic()-began)>a.max_hours*3600:break
            packet,spans,slots=source_for_case(case['question'],cid,out/'frozen_sources',
                    resolver,ctrl,harvester,a.db)
            order=['direct','verbatim'];random.Random(a.seed+sum(map(ord,cid))).shuffle(order)
            for arm in order:
                if (cid,arm) in seen:continue
                begin=time.monotonic();row={'case_id':cid,'arm':arm,'status':'ok','route':packet['route'],
                  'source_evidence_ids':[s.evidence_id for s in spans],
                  'source_snapshot':f'frozen_sources/{cid}.json','timestamp':datetime.now(timezone.utc).isoformat()}
                try:
                    if arm=='direct':row.update(baseline(llm,case['question'],spans,slots,a.max_calls))
                    else:
                        r=quote_probe.run(case['question'],spans)
                        row.update(r)
                except (ValueError,TypeError,KeyError,IndexError,OSError,RuntimeError) as e:
                    row['status']='error';row['error_type']=type(e).__name__;row['error_message']=str(e)[:300]
                row['wall_ms']=round((time.monotonic()-begin)*1000,1)
                with resfile.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                rows.append(row);seen.add((cid,arm))
                print(f'[{len(rows)}/16] {cid} {arm} {row["status"]} {row["wall_ms"]:.0f}ms '
                       f'candidates={row.get("accepted_count",row.get("accepted_quote_count",0))}',flush=True)
        result=summarize(rows)
        if len(rows)<16:result['state']='time_cap'
        write_atomic(out/'summary.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='../mhrag_gpu_chat_musa/data/knowledge.sqlite')
    p.add_argument('--cases',default='benchmarks/dev8_e2_quote_probe_NO_GOLD.json')
    p.add_argument('--aliases',default='study_aliases.reviewed.json')
    p.add_argument('--out',default='logs/v13_e2_quote_probe8_01')
    p.add_argument('--chat-url',default='http://127.0.0.1:7866')
    p.add_argument('--qwen-url',default='http://127.0.0.1:8002/v1')
    p.add_argument('--model',default='qwen-local')
    p.add_argument('--max-calls',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=0.75)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--report',action='store_true')
    a=p.parse_args()
    if a.report:
        path=Path(a.out)/'summary.json'
        if not path.is_file():print('No summary yet:',a.out);return
        print(path.read_text(encoding='utf-8'));return
    if not 1<=a.max_calls<=8 or not 0<a.max_hours<=2:raise ValueError('budget exceeded')
    cases=load_cases(a.cases)
    for pth in (a.db,a.aliases,a.cases):
        if not Path(pth).is_file():raise FileNotFoundError(pth)
    if not a.execute:
        print(json.dumps({'preflight':'OK','cases':8,'jobs':16,'arms':['direct','verbatim'],
              'max_llm_calls_per_arm_per_case':a.max_calls,
              'frozen_source_per_case':True,'gold_loaded':False},ensure_ascii=False,indent=2));return
    run(a)

if __name__=='__main__':main()
