#!/usr/bin/env python3
"""Offline CrossEncoder rerank with append-only per-query checkpointing.

Important: this measures document-level evidence, not original sentence-level recall.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sqlite3, sys, time
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mhrag_overnight_v3'))
from overnight_retrieval_eval_v2 import load_items, gold_keys, get_db


def eligible_ids(dataset):
    return [(i,str(row.get('question_type') or 'unknown')) for i,row in load_items(dataset) if gold_keys(row)]


def sample_ids(dataset,n,seed):
    rows=eligible_ids(dataset)
    if n<=0 or n>=len(rows):return set(i for i,_ in rows)
    rng=random.Random(seed)
    group=defaultdict(list)
    for i,t in rows:group[t].append(i)
    # Stratified deterministic without using answers or retriever performance.
    selected=[]
    quotas={typ:n*len(vals)/len(rows) for typ,vals in group.items()}
    counts={t:int(q) for t,q in quotas.items()}
    for t in sorted(group):rng.shuffle(group[t])
    left=n-sum(counts.values())
    for t in sorted(group,key=lambda x:(-(quotas[x]-counts[x]),x))[:left]:counts[t]+=1
    for t in sorted(group):selected.extend(group[t][:counts[t]])
    assert len(selected)==n
    return set(selected)


def load_rank_rows(path,indices,pool):
    out=[]
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if r['index'] not in indices:continue
            if pool not in r:raise ValueError(f'Pool {pool} absent in saved ranks')
            out.append((r['index'],list(dict.fromkeys(map(int,r[pool])))))
    out.sort()
    if not out:raise RuntimeError(f'No ranking entries in {path} matching selected indices')
    return out


def file_sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def recorded_ids(path):
    ids=set()
    if path.exists():
        with path.open(encoding='utf8') as f:
            for line in f:
                try:
                    r=json.loads(line)
                    if len(r['chunk_ids'])==len(r['scores']):ids.add(r['index'])
                except (ValueError,KeyError,TypeError):continue
    return ids


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ranks',required=True)
    p.add_argument('--db-path',default='mhrag_chat_mvp/data/knowledge.db')
    p.add_argument('--dataset',default='dataset/MultiHopRAG.json')
    p.add_argument('--pool',choices=('hybrid','bm25','dense'),required=True)
    p.add_argument('--sample',type=int,default=0,help='0 means all labeled questions; alternatives use same stratified 600')
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--top',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-length',type=int,default=512)
    p.add_argument('--model',required=True,help='Local CrossEncoder snapshot directory, NO remote model IDs')
    p.add_argument('--out',required=True)
    p.add_argument('--mock-scoring',action='store_true',help='TEST ONLY: lexical score for offline unit checks; never for research metrics')
    a=p.parse_args()
    if a.top<10 or a.top>100:raise ValueError('--top must be between 10 and 100')
    if not a.mock_scoring and (not Path(a.model).is_dir() or not (Path(a.model)/'config.json').is_file()):
        raise RuntimeError(f'Model must be a downloaded local directory: {a.model}')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ranks=Path(a.ranks).resolve(); indices=sample_ids(a.dataset,a.sample,a.seed)
    rows=load_rank_rows(ranks,indices,a.pool)
    params={'ranks_sha256':file_sha256(ranks),'db_path':str(Path(a.db_path).resolve()),'dataset':str(Path(a.dataset).resolve()),
            'pool':a.pool,'sample':a.sample,'seed':a.seed,'top':a.top,'model':str(Path(a.model).resolve()),
            'max_length':a.max_length,'mock_scoring':a.mock_scoring,'selected_ids':[i for i,_ in rows]}
    meta=out/'parameters.json';scores_path=out/'rerank_scores.jsonl'
    if meta.exists():
        prev=json.loads(meta.read_text())
        if prev!=params:raise RuntimeError('Changed model/sample/ranks/pool on resumed run. Use NEW --out directory.')
    else:
        if scores_path.exists():raise RuntimeError('Scores file without params: use new output directory')
        meta.write_text(json.dumps(params,indent=2,ensure_ascii=False),encoding='utf8')
    done=recorded_ids(scores_path)
    pending=[(i,ids[:a.top]) for i,ids in rows if i not in done]
    print(f'pool={a.pool}; selected={len(rows)}; checkpointed={len(rows)-len(pending)}; to_score={len(pending)}',flush=True)
    if not pending:
        (out/'DONE').write_text('complete\n')
        print('DONE',out,flush=True);return
    source={i:row['query'] for i,row in load_items(a.dataset) if i in indices}
    conn=get_db(a.db_path)
    needed={cid for _,ids in pending for cid in ids}
    texts={}
    # Fetch in bounded batches to avoid SQLite variable-count limits.
    ids_sorted=sorted(needed)
    for j in range(0,len(ids_sorted),500):
        part=ids_sorted[j:j+500]
        if not part:continue
        sql='SELECT id,body FROM chunks WHERE id IN ('+','.join('?'*len(part))+')'
        for r in conn.execute(sql,part):texts[int(r['id'])]=str(r['body'])
    conn.close()
    missing=needed-set(texts)
    if missing:raise RuntimeError(f'{len(missing)} retrieved chunk IDs not present in DB')
    model=None
    if not a.mock_scoring:
        import torch
        import torch_musa
        if not torch.musa.is_available():raise RuntimeError('CrossEncoder stage requires MUSA')
        from sentence_transformers import CrossEncoder
        model=CrossEncoder(a.model,device='musa',max_length=a.max_length,trust_remote_code=False)
        print(f'GPU {torch.musa.get_device_name(0)} model loaded from {a.model}',flush=True)
    start=time.monotonic()
    with scores_path.open('a',encoding='utf8') as f:
        for step,(i,ids) in enumerate(pending,1):
            q=source[i]
            if a.mock_scoring:
                qwords=set(q.lower().split())
                vals=[float(len(qwords&set(texts[c].lower().split()))) for c in ids]
            else:
                pairs=[(q,texts[c]) for c in ids]
                vals=[float(x) for x in model.predict(pairs,batch_size=a.batch_size,show_progress_bar=False)]
            f.write(json.dumps({'index':i,'pool':a.pool,'chunk_ids':ids,'scores':vals},ensure_ascii=False)+'\n')
            f.flush()
            if step%20==0:os.fsync(f.fileno())
            if step%25==0:print(f'Completed {step}/{len(pending)} remaining; seconds={time.monotonic()-start:.1f}',flush=True)
        os.fsync(f.fileno())
    if len(recorded_ids(scores_path))==len(rows):
        (out/'DONE').write_text('complete\n')
        print(f'DONE {out}',flush=True)
    else:raise RuntimeError('Some expected rows missing from checkpoint')

if __name__=='__main__':main()
