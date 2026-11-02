"""Developer-only 16-question gate, not a medical accuracy benchmark.

Gold labels are intentionally not read. Runs only after explicit --execute.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from datetime import datetime,timezone
from mhrag_v13.resolve import SQLiteStudyResolver
from mhrag_v13.retrieve import LocalChatRetriever
from mhrag_v13.grounded_answers import QwenJsonClient,ClaimDraftGenerator,ClaimSupportVerifier
from mhrag_v13.grounded_controller import GroundedController

PRIORITY=('v9_032','v9_042','v9_089','v9_090','v9_091','v9_096','v9_097',
          'compare_083_084_not_comparable','compare_081_085_autoimmune',
          'catalog_101','catalog_109','catalog_111','new_104_jcm_identify',
          'challenge_080_082_no_effect_size')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',required=True)
    p.add_argument('--cases',required=True,help='Frozen DEV questions; gold fields ignored')
    p.add_argument('--aliases')
    p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=16)
    p.add_argument('--mode',choices=('evidence','qwen'),default='qwen')
    p.add_argument('--judge',action='store_true')
    p.add_argument('--execute',action='store_true',help='Actually call local GPU services')
    p.add_argument('--max-hours',type=float,default=1.5)
    args=p.parse_args()
    if args.limit<1 or args.limit>30 or args.max_hours<=0 or args.max_hours>2:
        p.error('pilot bounds: 1..30 cases and 0..2 hours')
    if args.mode=='evidence' and args.judge:p.error('--judge requires qwen')
    raw=json.loads(Path(args.cases).read_text(encoding='utf-8'))
    cases=raw.get('cases',raw) if isinstance(raw,dict) else raw
    eligible={x['case_id']: {'case_id':x['case_id'],'question':x['question']}
              for x in cases if x.get('phase')=='baseline' and x.get('question') and x.get('case_id')}
    ordered=[eligible[k] for k in PRIORITY if k in eligible]
    ordered.extend(x for k,x in sorted(eligible.items()) if k not in PRIORITY)
    ordered=ordered[:args.limit]
    if not ordered:raise SystemExit('No baseline DEV cases')
    if not args.execute:
        print(json.dumps({'dry_run':True,'count':len(ordered),'case_ids':[x['case_id'] for x in ordered],
                          'gold_is_not_read':True},ensure_ascii=False,indent=2))
        return
    out=Path(args.out).resolve()
    if out.exists():raise SystemExit(f'Output exists; use fresh directory: {out}')
    if not Path(args.db).is_file():raise SystemExit('DB not found')
    out.mkdir(parents=True)
    project_root=Path(__file__).resolve().parents[1]
    lock_path=project_root.parent/'results'/'.v124_eval.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another RAG evaluation is running; refusing overlap')
        llm=QwenJsonClient() if args.mode=='qwen' else None
        runner=GroundedController(resolver=SQLiteStudyResolver(args.db,args.aliases),
            retriever=LocalChatRetriever(),db_path=args.db,
            generator=ClaimDraftGenerator(llm) if llm else None,
            claim_verifier=ClaimSupportVerifier(llm if args.judge else None))
        start=time.monotonic();success=0;fail=0
        md=hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()
        (out/'manifest.json').write_text(json.dumps({'mode':args.mode,'judge':args.judge,
            'cases_sha256':md,'db':str(Path(args.db).resolve()),
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'labels_used_for_retrieval':False},indent=2),encoding='utf-8')
        with (out/'results.jsonl').open('w',encoding='utf-8') as f:
            for i,row in enumerate(ordered,1):
                if (time.monotonic()-start)>args.max_hours*3600:
                    print('TIME_CAP; keeping partial results',flush=True);break
                try:
                    result=runner.run(row['question'],evidence_only=(args.mode=='evidence'))
                    item={'case_id':row['case_id'],'status':'ok','result':result}
                    success+=1
                    print(f'[{i}/{len(ordered)}] ok {row["case_id"]} decision={result["decision"]["decision"]} accepted={result["accepted_claim_count"]}',flush=True)
                except Exception as exc:
                    fail+=1
                    item={'case_id':row['case_id'],'status':'error','error_type':type(exc).__name__}
                    print(f'[{i}/{len(ordered)}] ERROR {row["case_id"]} {type(exc).__name__}',flush=True)
                    if fail>=3:
                        f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());break
                f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        summary={'planned':len(ordered),'completed':success,'errors':fail,
                 'wall_seconds':round(time.monotonic()-start,1),'not_medical_accuracy':True}
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('END:',summary,flush=True)

if __name__=='__main__':main()
