"""CLI: python -m mhrag_v13.grounded_cli --db ... --question ..."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .resolve import SQLiteStudyResolver
from .retrieve import LocalChatRetriever
from .evidence_slots import SQLiteEvidenceHarvester
from .grounded_answers import QwenJsonClient,ClaimDraftGenerator,ClaimSupportVerifier
from .grounded_controller import GroundedController
from .tracing import JsonlTrace

def main():
    p=argparse.ArgumentParser(description='MultiHop-RAG v13 isolated evidence closure prototype')
    p.add_argument('--db',required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--aliases')
    p.add_argument('--chat-url',default='http://127.0.0.1:7866')
    p.add_argument('--qwen-url',default='http://127.0.0.1:8002/v1')
    p.add_argument('--model',default='qwen-local')
    p.add_argument('--mode',choices=('evidence','qwen'),default='evidence')
    p.add_argument('--judge',action='store_true',help='Second independent Qwen evidence support pass (not clinical gold)')
    p.add_argument('--max-http',type=int,default=4)
    p.add_argument('--max-draft-calls',type=int,default=6,help='Maximum bounded local Qwen claim-drafting requests')
    p.add_argument('--draft-strategy',choices=('text-first','slots-first','batch-recover'),default='text-first',
                   help='For Qwen 7B use text-first without JSON/ID copying; legacy strategies remain available')
    p.add_argument('--trace-file')
    args=p.parse_args()
    db=Path(args.db).resolve()
    if not db.is_file():p.error('DB not found')
    if args.mode=='evidence' and args.judge:p.error('--judge requires --mode qwen')
    client=QwenJsonClient(args.qwen_url,args.model) if args.mode=='qwen' else None
    ctrl=GroundedController(resolver=SQLiteStudyResolver(db,args.aliases),
         retriever=LocalChatRetriever(args.chat_url),db_path=db,
         generator=ClaimDraftGenerator(client,max_calls=args.max_draft_calls,strategy=args.draft_strategy) if client else None,
         claim_verifier=ClaimSupportVerifier(client if args.judge else None),
         trace=JsonlTrace(args.trace_file) if args.trace_file else None,
         max_http_calls=args.max_http)
    result=ctrl.run(args.question,evidence_only=(args.mode=='evidence'))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
