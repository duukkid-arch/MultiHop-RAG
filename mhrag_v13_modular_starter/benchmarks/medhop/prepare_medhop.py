#!/usr/bin/env python3
"""Convert original QAngaroo MedHop to per-case context-corpus and blind QA inputs.

No GPU, no dependencies, no modification to original DB. Gold labels isolated.
This is a closed-context multihop reading-comprehension protocol, NOT an
open-corpus RAG benchmark or a human/PDF-verified new holdout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def parse_records(path, split):
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError(f'{split}: expected nonempty JSON array')
    used = set()
    questions, corpus, gold = [], [], []
    for i, r in enumerate(records):
        for key in ('id', 'query', 'candidates', 'supports', 'answer'):
            if key not in r:
                raise ValueError(f'{split} row {i} missing {key}')
        rid = str(r['id']).strip()
        if not rid or rid in used:
            raise ValueError(f'{split} row {i} empty/duplicate id: {rid}')
        used.add(rid)
        query = r['query']
        candidates = r['candidates']
        supports = r['supports']
        answer = r['answer']
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f'{rid}: invalid query')
        if not isinstance(candidates, list) or len(candidates) < 2 or not all(isinstance(s,str) and s.strip() for s in candidates):
            raise ValueError(f'{rid}: invalid candidates')
        if len(set(candidates)) != len(candidates) or answer not in candidates:
            raise ValueError(f'{rid}: duplicate candidates or gold not among candidates')
        if not isinstance(supports, list) or not supports or not all(isinstance(s,str) and s.strip() for s in supports):
            raise ValueError(f'{rid}: invalid supports')
        case_id = f'medhop_{split}_{rid}'
        questions.append({'case_id':case_id, 'question':query, 'candidates':candidates,
                          'corpus_scope':case_id, 'source_split':split})
        for j, doc in enumerate(supports):
            corpus.append({'doc_id':f'{case_id}_doc{j:04d}', 'corpus_scope':case_id,
                           'text':doc, 'source_split':split})
        gold.append({'case_id':case_id, 'answer':answer, 'source_split':split})
    return used, questions, corpus, gold


def prepare(data_dir, out):
    p1, p2 = data_dir / 'train.json', data_dir / 'dev.json'
    if not p1.is_file() or not p2.is_file():
        raise FileNotFoundError('Need official qangaroo_v1.1/medhop/train.json and dev.json')
    if out.exists():
        raise FileExistsError(f'output already exists, refusing overwrite: {out}')
    sets = {}
    for split, path in [('train',p1),('dev',p2)]:
        sets[split] = parse_records(path, split)
    if sets['train'][0] & sets['dev'][0]:
        raise ValueError('train/dev ID overlap')
    out.mkdir(parents=True)
    counts = {}
    for split in ('train','dev'):
        _, qs, docs, gold = sets[split]
        write_jsonl(out/f'{split}_questions_NO_GOLD.jsonl', qs)
        write_jsonl(out/f'{split}_context_corpus.jsonl', docs)
        write_jsonl(out/f'{split}_gold_PRIVATE.jsonl', gold)
        counts[split]={'questions':len(qs),'support_documents':len(docs)}
    manifest = {
        'dataset':'QAngaroo MedHop original',
        'benchmark_type':'question_scoped_multi_document_reasoning',
        'warning':'Gold is publicly released; dev is public validation, NOT private sealed holdout',
        'corpus_policy':'Per-case support documents only. Do not insert all case corpora into existing 116-PDF index.',
        'model_input_fields':['case_id','question','candidates','corpus_scope'],
        'source_sha256':{'train.json':sha256(p1),'dev.json':sha256(p2)},
        'counts':counts,
        'license':'CC BY-SA 3.0; keep dataset attribution and license obligations',
    }
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'READY_NO_GPU', 'out':str(out), **counts},ensure_ascii=False))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True,help='directory containing original MedHop train.json and dev.json')
    p.add_argument('--out',type=Path,required=True,help='must not exist')
    a=p.parse_args()
    try: prepare(a.data_dir,a.out)
    except (OSError,ValueError,TypeError) as e:
        sys.exit('REFUSED: '+str(e))
