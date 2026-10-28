#!/usr/bin/env python3
"""Actual MUSA CE inference time for K=60 vs K=100, warm model, same stratified questions.
Never run concurrently with v6. Results append per query, interruption-safe.
"""
import argparse, hashlib, json, os, random, sqlite3, statistics, sys, time
from pathlib import Path

pa=argparse.ArgumentParser()
pa.add_argument('--project',default='/root/autodl-tmp/MultiHop-RAG')
pa.add_argument('--v6-root',default='results/v6_focus_750_musa')
pa.add_argument('--out',default='results/v7_8h_budget/latency')
pa.add_argument('--sample',type=int,default=40)
pa.add_argument('--batch-size',type=int,default=4)
pa.add_argument('--max-length',type=int,default=512)
pa.add_argument('--max-seconds',type=int,default=2700)
pa.add_argument('--seed',type=int,default=20261027)
a=pa.parse_args()
project=Path(a.project).resolve();os.chdir(project)
sys.path.insert(0,str(project/'mhrag_v6_musa'))
from rerank_musa import load_items, sample_ids, load_rank_rows, get_db
import torch,torch_musa
from sentence_transformers import CrossEncoder

v6=Path(a.v6_root);v6=v6 if v6.is_absolute() else project/v6
out=Path(a.out);out=out if out.is_absolute() else project/out
out.mkdir(parents=True,exist_ok=True)
if (v6/'STATUS').read_text().strip()!='DONE':
    raise SystemExit('V6 not DONE; do not benchmark concurrently with GPU scoring.')
ranks=v6/'rechunk750/hybrid_bge/retrieval_ranks.jsonl'
db=v6/'rechunk750/corpus.db'
dataset=project/'dataset/MultiHopRAG.json'
model_dir=Path('/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')
indices=sample_ids(str(dataset),a.sample,a.seed)
rows=load_rank_rows(ranks,indices,'hybrid')
queries={i:row['query'] for i,row in load_items(str(dataset)) if i in indices}
needed={cid for _,ids in rows for cid in ids[:100]}
conn=get_db(str(db));texts={}
ids_sorted=sorted(needed)
for j in range(0,len(ids_sorted),500):
    sub=ids_sorted[j:j+500]
    sql='SELECT id,body FROM chunks WHERE id IN ('+','.join('?'*len(sub))+')'
    for r in conn.execute(sql,sub):texts[int(r['id'])]=str(r['body'])
conn.close()
missing=needed-set(texts)
if missing:raise RuntimeError(f'Missing {len(missing)} chunks')
path=out/'latency.jsonl'
meta={'sample':a.sample,'batch_size':a.batch_size,'max_length':a.max_length,'seed':a.seed,
      'ranks_sha256':hashlib.sha256(ranks.read_bytes()).hexdigest(),'model_dir':str(model_dir)}
mp=out/'params.json'
if mp.exists():
    if json.loads(mp.read_text())!=meta:raise RuntimeError('Benchmark configuration changed: use a new --out')
else:
    if path.exists():raise RuntimeError('Existing latency.jsonl without params.json')
    mp.write_text(json.dumps(meta,indent=2))
done=set();observed=[]
if path.exists():
    for line in path.read_text().splitlines():
        try:
            r=json.loads(line);done.add((r['index'],r['k']));observed.append(r)
        except (ValueError,KeyError):pass
if not torch.musa.is_available():raise RuntimeError('MUSA unavailable')
model=CrossEncoder(str(model_dir),device='musa',max_length=a.max_length,trust_remote_code=False)
# Warmup excluded from timing.
first_i,first_ids=rows[0]
model.predict([(queries[first_i],texts[c]) for c in first_ids[:min(4,len(first_ids))]],batch_size=a.batch_size,show_progress_bar=False)
if hasattr(torch.musa,'synchronize'):torch.musa.synchronize()
start=time.monotonic()
print('READY',len(rows),'questions; remaining pairs',sum((i,k) not in done for i,_ in rows for k in (60,100)),flush=True)
with path.open('a',encoding='utf-8') as f:
    for count,(i,ids) in enumerate(rows):
        order=(60,100) if count%2==0 else (100,60)
        for k in order:
            if (i,k) in done:continue
            if time.monotonic()-start>=a.max_seconds:
                print('TIME_LIMIT reached; partial results checkpointed',flush=True)
                break
            pairs=[(queries[i],texts[c]) for c in ids[:k]]
            if hasattr(torch.musa,'synchronize'):torch.musa.synchronize()
            t=time.monotonic()
            scores=model.predict(pairs,batch_size=a.batch_size,show_progress_bar=False)
            if hasattr(torch.musa,'synchronize'):torch.musa.synchronize()
            sec=time.monotonic()-t
            rec={'index':i,'k':k,'scored_pairs':len(pairs),'seconds':round(sec,6),'returned':len(scores)}
            f.write(json.dumps(rec)+'\n');f.flush();observed.append(rec)
            print('MEASURED',i,'k',k,'seconds',f'{sec:.3f}',flush=True)
        if time.monotonic()-start>=a.max_seconds:break
for k in (60,100):
    vals=[x for x in observed if x['k']==k]
    if vals:
        print('SUMMARY',k,'queries',len(vals),'median_s',round(statistics.median(x['seconds'] for x in vals),3),
              'pairs_per_s',round(sum(x['scored_pairs'] for x in vals)/sum(x['seconds'] for x in vals),2),flush=True)
print('SAVED',path)
