#!/usr/bin/env python3
"""E2 field-first DEV8 regression, comparing to saved historic direct answers.

Consumes frozen source snapshots and direct-arm results already on AutoDL;
no retrieval, no SQLite writes, no Gold or sealed-holdout access.
"""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,os,statistics,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE_DIR=ROOT/'logs/v13_e2_quote_probe8_01'
OUT=ROOT/'logs/v13_e2_field8_01'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(65536),b''):h.update(b)
    return h.hexdigest()


def atomic(path,stuff):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(stuff,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def resources():
    from mhrag_v13.evidence_slots import EvidenceSpan
    cases_path=ROOT/'benchmarks/dev8_e2_quote_probe_NO_GOLD.json'
    d=json.loads(cases_path.read_text(encoding='utf-8'))
    assert d.get('dataset_kind')=='historical_dev_8_prompt_diagnostic_no_gold'
    cases=d['cases']
    if len(cases)!=8 or len({x['case_id'] for x in cases})!=8:raise RuntimeError('requires exactly 8 unique DEV cases')
    for c in cases:
        if set(c)-{'case_id','question','case_type'}:raise RuntimeError('gold/unsupported field in cases')
    old=SOURCE_DIR/'results.jsonl';summary=SOURCE_DIR/'summary.json'
    if not (old.is_file() and summary.is_file()):raise RuntimeError('old quote probe run missing or incomplete')
    if json.loads(summary.read_text(encoding='utf-8')).get('state')!='complete':raise RuntimeError('old DEV8 run not complete')
    older=[json.loads(x) for x in old.read_text(encoding='utf-8').splitlines() if x.strip()]
    olds={r['case_id']:r for r in older if r.get('arm')=='direct' and r.get('status')=='ok'}
    if set(olds)!={c['case_id'] for c in cases}:raise RuntimeError('old direct arm does not cover all DEV8 cases')
    packets={};spans={}
    for c in cases:
        path=SOURCE_DIR/'frozen_sources'/f"{c['case_id']}.json"
        if not path.is_file():raise RuntimeError('missing frozen source '+str(path))
        packet=json.loads(path.read_text(encoding='utf-8'))
        if packet.get('question')!=c['question'] or not packet.get('no_gold'):
            raise RuntimeError('source question mismatch or gold contamination')
        packets[c['case_id']]=packet
        spans[c['case_id']]=[EvidenceSpan(**s) for s in packet['spans']]
    paths=[cases_path,old,summary,ROOT/'mhrag_v13/e2_field8.py',ROOT/'benchmarks/run_e2_field8.py']
    paths +=[SOURCE_DIR/'frozen_sources'/f"{c['case_id']}.json" for c in cases]
    config={str(p.relative_to(ROOT)):digest(p) for p in paths}
    return cases,olds,packets,spans,config


def summarize(cases,baseline,rows):
    from mhrag_v13.e2_field8 import plan,field_answer_hit
    byid={r['case_id']:r for r in rows}
    historical=Counter();new=Counter()
    old_full=0;new_full=0;old_any=0;new_any=0;miss=[]
    for case in cases:
        cid=case['case_id'];fields=plan(case['question'])[1]
        o=baseline[cid]
        old_accepted=o.get('accepted') or []
        oldtext=' '.join(x.get('text','') for x in old_accepted)
        old_cov=[f.name for f in fields if field_answer_hit(f,oldtext)]
        if old_accepted:old_any+=1
        if len(old_cov)==len(fields):old_full+=1
        if cid in byid:
            r=byid[cid]
            if r.get('accepted_count'):new_any+=1
            if len(r.get('covered_fields',[]))==len(fields):new_full+=1
            for d in r.get('diagnostics',[]):new[d['status']]+=1
        else:miss.append(cid)
        for d in o.get('diagnostics',[]):historical[d['status']]+=1
    complete=len(rows)==len(cases) and all(r.get('status')=='ok' for r in rows)
    return {'state':'complete' if complete else 'incomplete','planned':len(cases),'completed':len(rows),
       'runtime_errors':sum(r.get('status')!='ok' for r in rows),
       'historical_direct':{'cases_with_accepted_claim':old_any,
                'automatic_field_cues_all_present':old_full,'diagnostic_events':dict(historical)},
       'new_field_first':{'cases_with_accepted_claim':new_any,
                'automatic_field_cues_all_present':new_full,'diagnostic_events':dict(new),
                'total_generation_calls':sum(r.get('generation_calls',0) for r in rows),
                'median_wall_ms':round(statistics.median([r['wall_ms'] for r in rows]),1) if rows else None},
       'missing_cases':miss,'comparison':'historic direct-arm replay vs new live field-first on IDENTICAL frozen spans; NOT simultaneous randomized arms',
       'evaluation':'DEV8 reused for design, no sealed holdout; field coverage is heuristic; medical accuracy NOT_EVALUATED',
       'medical_accuracy':'NOT_EVALUATED'}


def export_review(cases,older,rows):
    outfile=OUT/'review_for_human.csv';byid={r['case_id']:r for r in rows}
    if not all(c['case_id'] in byid for c in cases):return
    with outfile.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','question','historical_direct_candidate','field_first_candidate',
           'field_first_covered_fields','field_first_missing_fields','field_first_evidence_ids',
           'relevance_human','source_support_human','medical_accuracy_human','comments_human'])
        writer.writeheader()
        for c in cases:
            old=older[c['case_id']];r=byid[c['case_id']]
            writer.writerow({'case_id':c['case_id'],'question':c['question'],
                'historical_direct_candidate':' | '.join(x.get('text','') for x in old.get('accepted',[])),
                'field_first_candidate':' | '.join(x.get('text','') for x in r.get('accepted',[])),
                'field_first_covered_fields':','.join(r.get('covered_fields',[])),
                'field_first_missing_fields':','.join(r.get('missing_fields',[])),
                'field_first_evidence_ids':','.join(x.get('evidence_id','') for x in r.get('accepted',[]))})
    print('REVIEW_EXPORT_OK',outfile)


def run(a):
    from mhrag_v13.grounded_answers import QwenJsonClient
    from mhrag_v13.e2_field8 import field_first_generate,rank_spans,plan
    cases,older,packets,spans,hashes=resources()
    config={'hashes':hashes,'model':a.model,'qwen_url':a.qwen_url,'max_calls':a.max_calls}
    if a.resume:
        if not (OUT/'manifest.json').is_file():raise RuntimeError('resume requires existing manifest')
        old=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        if old.get('config')!=config:raise RuntimeError('RESUME REFUSED: frozen source/code/model changed')
    else:
        if OUT.exists():raise RuntimeError('result directory exists; refusing overwrite')
        OUT.mkdir(parents=True)
        atomic(OUT/'manifest.json',{'config':config,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'design':'HISTORIC_DEV8_LOCKED_FROZEN_SPANS_NO_GOLD'})
    lock=ROOT.parent/'results'/'.v124_eval.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as lk:
        try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('other GPU evaluation holds lock')
        llm=QwenJsonClient(a.qwen_url,a.model)
        # Ping local model endpoint before first Qwen generation.
        import urllib.request
        with urllib.request.urlopen(a.qwen_url.rstrip('/')+'/models',timeout=12) as rsp:
            if rsp.status!=200:raise RuntimeError('Qwen endpoint unavailable')
        fp=OUT/'results.jsonl';rows=[json.loads(s) for s in fp.read_text(encoding='utf-8').splitlines() if s.strip()] if fp.exists() else []
        seen={r['case_id'] for r in rows}
        if len(seen)!=len(rows):raise RuntimeError('duplicate case results')
        start=time.monotonic()
        for c in cases:
            cid=c['case_id']
            if cid in seen:continue
            if time.monotonic()-start>a.max_hours*3600:break
            begin=time.monotonic()
            row={'case_id':cid,'question':c['question'],'status':'ok','timestamp_utc':datetime.now(timezone.utc).isoformat()}
            try:
                row.update(field_first_generate(llm,c['question'],spans[cid],a.max_calls))
                row['source_span_count']=len(spans[cid])
                row['source_snapshot_path']=f'logs/v13_e2_quote_probe8_01/frozen_sources/{cid}.json'
            except (ValueError,RuntimeError,OSError,IndexError,TypeError,KeyError) as exc:
                row.update({'status':'error','error_type':type(exc).__name__,'error_message':str(exc)[:280]})
            row['wall_ms']=round(1000*(time.monotonic()-begin),1)
            with fp.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            rows.append(row);seen.add(cid)
            print(f"[{len(rows)}/8] {cid} {row['status']} accepted={row.get('accepted_count',0)} "
                  f"fields={row.get('covered_fields',[])} calls={row.get('generation_calls',0)}",flush=True)
        report=summarize(cases,older,rows)
        if len(rows)<8:report['state']='time_cap'
        atomic(OUT/'summary.json',report)
        if report['state']=='complete':export_review(cases,older,rows)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--qwen-url',default='http://127.0.0.1:8002/v1')
    p.add_argument('--model',default='qwen-local')
    p.add_argument('--max-calls',type=int,default=4)
    p.add_argument('--max-hours',type=float,default=0.75)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--report',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.report:
        path=OUT/'summary.json'
        if not path.is_file():print('No summary yet:',OUT.relative_to(ROOT));return
        print(path.read_text(encoding='utf-8'));return
    if a.max_calls!=4:raise SystemExit('fixed design requires --max-calls=4')
    if not a.qwen_url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise SystemExit('only local Qwen endpoint permitted')
    cases,old,packets,spans,hashes=resources()
    from mhrag_v13.e2_field8 import plan,rank_spans
    report={'preflight':'OK','cases':len(cases),'new_gpu_tasks':len(cases),
         'reuse_old_direct_no_gpu':True,'max_llm_calls_per_case':4,
         'source_ranking_field_signals_TOP1':0,'rankings':[]}
    for c in cases:
        fields=plan(c['question'])[1];ranked=rank_spans(c['question'],spans[c['case_id']])
        first=ranked[0]
        fully=set(first['fields'])=={f.name for f in fields}
        report['source_ranking_field_signals_TOP1']+=fully
        report['rankings'].append({'case_id':c['case_id'],'top1_id':first['span'].evidence_id,
           'old_rank':first['source_rank'],'fields_in_top1':first['fields'],
           'expected_fields':[f.name for f in fields]})
    if not a.execute:
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    run(a)

if __name__=='__main__':main()
