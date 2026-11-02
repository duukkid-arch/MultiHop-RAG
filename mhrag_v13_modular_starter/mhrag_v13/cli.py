"""Standalone CLI: python -m mhrag_v13.cli --db ... --question ..."""
from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
from .resolve import SQLiteStudyResolver
from .retrieve import LocalChatRetriever
from .verify import StructuralEvidenceGate
from .answer import EvidencePacketAnswerer
from .qwen_answer import LocalQwenAnswerer
from .open_document import SQLiteSectionReader
from .controller import AdaptiveController
from .tracing import JsonlTrace

def main():
    parser=argparse.ArgumentParser(description='MultiHop-RAG v13 modular research sidecar')
    parser.add_argument('--db',required=True,help='Read-only knowledge.sqlite path')
    parser.add_argument('--chat-url',default='http://127.0.0.1:7866')
    parser.add_argument('--aliases',help='Optional human-reviewed study alias registry JSON')
    parser.add_argument('--resolve-only',action='store_true',help='Inspect study identification without GPU/API calls')
    parser.add_argument('--question',help='Query; omit with --doctor')
    parser.add_argument('--answerer',choices=['evidence','qwen'],default='evidence')
    parser.add_argument('--max-http',type=int,default=4)
    parser.add_argument('--trace-file',help='Local JSONL path; no raw prompt/source included')
    parser.add_argument('--doctor',action='store_true',help='Only test local corpus schema (no model calls)')
    args=parser.parse_args()
    db=Path(args.db).resolve()
    if not db.is_file():parser.error(f'DB not found: {db}')
    if args.doctor:
        with sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True) as con:
            counts=[con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('documents','chunks')]
            chk=con.execute('pragma quick_check').fetchone()[0]
        print(json.dumps({'documents':counts[0],'chunks':counts[1],'quick_check':chk},indent=2))
        return
    if not args.question:parser.error('--question required unless --doctor')
    resolver=SQLiteStudyResolver(db,args.aliases)
    if args.resolve_only:
        res=resolver.resolve(args.question)
        print(json.dumps({'required_ids':list(res.ids),
            'unresolved_anchors':list(res.unresolved), 'complete':res.complete,
            'candidate_scores':{anchor:resolver.candidates(anchor) for anchor in res.anchors},
            'resolved': [{'document_id':st.document_id, 'title':st.title,
                          'anchor':st.anchor, 'score':st.confidence} for st in res.studies]},
            ensure_ascii=False,indent=2))
        return
    controller=AdaptiveController(resolver, LocalChatRetriever(args.chat_url),
       StructuralEvidenceGate(), LocalQwenAnswerer() if args.answerer=='qwen' else EvidencePacketAnswerer(),
       section_reader=SQLiteSectionReader(db),
       trace=JsonlTrace(args.trace_file) if args.trace_file else None,max_http_calls=args.max_http)
    print(json.dumps(controller.run(args.question),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
