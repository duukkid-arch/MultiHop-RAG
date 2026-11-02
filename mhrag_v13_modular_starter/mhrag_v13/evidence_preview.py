"""Read-only audit of the exact text-first model evidence windows (no Qwen or chat API)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .resolve import SQLiteStudyResolver
from .evidence_slots import SQLiteEvidenceHarvester
from .grounded_answers import ClaimDraftGenerator, QwenJsonClient


def main():
    p=argparse.ArgumentParser(description='Preview exactly which original SQLite spans would be shown to Qwen')
    p.add_argument('--db',required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--aliases')
    args=p.parse_args()
    db=Path(args.db).resolve()
    if not db.is_file():p.error('DB not found')
    identity=SQLiteStudyResolver(db,args.aliases).resolve(args.question)
    spans,slots=SQLiteEvidenceHarvester(db).harvest(identity.ids,args.question)
    generator=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'NO_EVIDENCE'),strategy='text-first')
    print(json.dumps({'identity':{'required_ids':list(identity.ids),'unresolved':list(identity.unresolved),
                                  'complete':identity.complete},
                      'selected_windows':generator.preview_evidence_windows(spans,slots),
                      'note':'read-only model context selection; NOT factual verification'},
                     ensure_ascii=False,indent=2))


if __name__=='__main__':main()
