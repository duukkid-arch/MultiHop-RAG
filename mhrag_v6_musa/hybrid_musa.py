#!/usr/bin/env python3
"""Safe offline eval for MultiHop-RAG MVP SQLite FTS5 and optional GPU BGE-M3 hybrid.

Measures GOLD DOCUMENT coverage (not original sentence-level Recall@10).
No LLM or hidden Chain-of-Thought. No changes to live database.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# App code is imported only after adding --project to sys.path.
def load_items(path, max_queries=0):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next((data[k] for k in ('data','questions','queries') if isinstance(data.get(k),list)), [])
    if not isinstance(data,list): raise ValueError('dataset must contain a list')
    out=[]
    for i, x in enumerate(data):
        if not isinstance(x, dict): continue
        q=x.get('query')
        if not isinstance(q,str) or not q.strip(): continue
        out.append((i,x))
        if max_queries and len(out)>=max_queries: break
    return out

def norm_title(s):
    return re.sub(r'[^a-z0-9\u3400-\u9fff]+',' ',str(s or '').casefold()).strip()

def norm_url(s):
    try:
        p=urlsplit(str(s or '').strip())
        if not p.netloc: return ''
        return (p.netloc.lower() + p.path.rstrip('/')).lower()
    except Exception: return ''

def gold_keys(row):
    keys=[]
    for e in row.get('evidence_list') or []:
        if not isinstance(e,dict): continue
        u=norm_url(e.get('url')); t=norm_title(e.get('title'))
        if u or t: keys.append((u,t))
    return list(dict.fromkeys(keys))

def doc_match(doc,gold):
    u,t=gold
    # Use URL where both sides have one; fallback to exact title only when URL absent.
    if u and doc['url_key']: return u==doc['url_key']
    return bool(t and t==doc['title_key'])

def get_db(db_path):
    if not Path(db_path).is_file(): raise FileNotFoundError(db_path)
    conn=sqlite3.connect(f'file:{Path(db_path).resolve()}?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    return conn

def chunks_and_docs(db):
    docs={}
    for d in db.execute('SELECT id,title,url FROM documents'):
        docs[d['id']]={'url_key':norm_url(d['url']), 'title_key':norm_title(d['title']), 'title':d['title'], 'id':d['id']}
    return docs

def unique_docs(chunk_ids, chunk_to_doc, limit=10):
    output=[]; seen=set()
    for ch in chunk_ids:
        did=chunk_to_doc.get(ch)
        if did is None or did in seen: continue
        seen.add(did); output.append(did)
        if len(output)>=limit: break
    return output

def fts_chunks(db, question, tokenize, limit=100):
    words=list(dict.fromkeys(tokenize(question)))[:24]
    if not words: return []
    match=' OR '.join('"'+w+'"' for w in words)
    return [int(row[0]) for row in db.execute(
        'SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? '
        'ORDER BY bm25(chunks_fts,2.8,1.0) LIMIT ?', (match, limit))]

def metric(doc_ids, docs, gold):
    if not gold: return {'covered':None,'total':0,'all':None,'hit':None}
    candidates=[docs[did] for did in doc_ids if did in docs]
    covered=sum(any(doc_match(d,g) for d in candidates) for g in gold)
    return {'covered':covered,'total':len(gold),'all':int(covered==len(gold)),'hit':int(covered>0)}

def rrf(*rank_lists, k=60):
    scores=defaultdict(float)
    for seq in rank_lists:
        for rank,chunk_id in enumerate(seq,1):
            scores[chunk_id]+=1/(k+rank)
    return sorted(scores,key=lambda ch:(-scores[ch],ch))

def ensure_dir(d): Path(d).mkdir(parents=True,exist_ok=True)

def summarize(rows):
    report={}
    for method in ('bm25','dense','hybrid','hybrid_bge'):
        available=[r for r in rows if r.get(method+'_all') is not None]
        if not available:continue
        out={'n':len(available),'gold_doc_coverage_at10':round(sum(r[method+'_covered']/r['gold_total'] for r in available)/len(available),6),
             'all_gold_docs_hit_at10':round(sum(r[method+'_all'] for r in available)/len(available),6),
             'at_least_one_gold_doc_at10':round(sum(r[method+'_hit'] for r in available)/len(available),6)}
        bytype={}
        for ty in sorted(set(r['question_type'] for r in available)):
            vals=[r for r in available if r['question_type']==ty]
            bytype[ty]={'n':len(vals),'gold_doc_coverage_at10':round(sum(r[method+'_covered']/r['gold_total'] for r in vals)/len(vals),6),'all_gold_docs_hit_at10':round(sum(r[method+'_all'] for r in vals)/len(vals),6)}
        out['by_type']=bytype;report[method]=out
    return report

def dump(rows, report, folder, extra=None):
    ensure_dir(folder)
    fields=['index','question_type','question','answer','gold_total','gold_indexable_count']
    for method in ('bm25','dense','hybrid','hybrid_bge'):
        fields.extend([method+'_'+k for k in ('covered','all','hit','document_ids','document_titles')])
    p=Path(folder)
    with (p/'per_query.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    with (p/'summary.json').open('w',encoding='utf-8') as f:
        json.dump({'metrics':report,'notes':'Document-level exact URL/title match; not sentence Recall@10; excludes queries without gold docs.', 'run':extra or {}},f,ensure_ascii=False,indent=2)
    with (p/'summary.txt').open('w',encoding='utf-8') as f:
        f.write('MultiHop-RAG offline GOLD-DOCUMENT matching metrics (NOT sentence-level Recall@10)\n')
        for method,x in report.items():
            f.write(f"{method}: n={x['n']}, coverage@10={x['gold_doc_coverage_at10']:.4f}, all_gold@10={x['all_gold_docs_hit_at10']:.4f}, any_gold@10={x['at_least_one_gold_doc_at10']:.4f}\n")
            for typ,v in x['by_type'].items():f.write(f"   {typ}: n={v['n']} coverage@10={v['gold_doc_coverage_at10']:.4f} all_gold@10={v['all_gold_docs_hit_at10']:.4f}\n")
        valid=[r for r in rows if r.get('gold_total',0)>0]
        if valid and all('gold_indexable_count' in r for r in valid):
            f.write(f'Original labelled gold document indexability: {sum(r["gold_indexable_count"] for r in valid)/max(1,sum(r["gold_total"] for r in valid)):.4f} (document-weighted; matching exact URL or title)\n')
        f.write('\nCaveat: Document presence is not evidence sentence completeness, nor answer accuracy.\n')

def encode_hybrid(model_name, chunks, questions, max_seq_length, batch_size):
    import torch
    import torch_musa
    from sentence_transformers import SentenceTransformer
    import numpy as np
    if not torch.musa.is_available(): raise RuntimeError('No MUSA GPU visible. Use --mode baseline instead.')
    print('MUSA:',torch.musa.get_device_name(0),flush=True)
    model=SentenceTransformer(model_name,device='musa',trust_remote_code=True)
    if hasattr(model,'max_seq_length'):model.max_seq_length=max_seq_length
    texts=[ch[1] for ch in chunks]
    print(f'Embedding {len(texts)} chunks',flush=True)
    corpus=model.encode(texts,batch_size=batch_size,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
    print(f'Embedding {len(questions)} queries',flush=True)
    qemb=model.encode(questions,batch_size=batch_size,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
    del model
    if torch.musa.is_available():torch.musa.empty_cache()
    return np.asarray(corpus,dtype='float32'), np.asarray(qemb,dtype='float32')

def dense_topk(corpus_emb,qemb,k=100,batch=64):
    """Use CUDA batched topk if available; CPU numpy fallback for small tests."""
    try:
        import torch
        import torch_musa
        if False:  # CPU NumPy for stable top-k on this small reference corpus
            device=torch.device('musa')
            corpus=torch.as_tensor(corpus_emb,device=device,dtype=torch.float32)
            query=torch.as_tensor(qemb,device=device,dtype=torch.float32)
            for left in range(0,len(query),batch):
                similarities=query[left:left+batch] @ corpus.T
                order=torch.topk(similarities,min(k,corpus.shape[0]),dim=1).indices.cpu().tolist()
                for ids in order:yield ids
            return
    except ImportError:pass
    import numpy as np
    for x in qemb:
        sims=corpus_emb@x
        cut=min(k,len(sims))
        if cut<=0:yield [];continue
        order=np.argpartition(sims,-cut)[-cut:]
        yield sorted(order,key=lambda j:(-float(sims[j]),int(j)))


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--project',default='mhrag_chat_mvp')
    pa.add_argument('--dataset',default='dataset/MultiHopRAG.json')
    pa.add_argument('--db-path',default='',help='Use copied or scaled SQLite database; NEVER change the live DB')
    pa.add_argument('--sample-queries',type=int,default=0,help='Random deterministic sample; independent of --max-queries')
    pa.add_argument('--seed',type=int,default=20261027)
    pa.add_argument('--out',default='results/chat_retrieval_overnight')
    pa.add_argument('--mode',choices=['baseline','hybrid','hybrid_bge'],default='baseline')
    pa.add_argument('--max-queries',type=int,default=0)
    pa.add_argument('--embedding-model',default='BAAI/bge-m3')
    pa.add_argument('--rerank-model',default='BAAI/bge-reranker-v2-m3')
    pa.add_argument('--batch-size',type=int,default=8)
    pa.add_argument('--max-seq-length',type=int,default=512)
    pa.add_argument('--rerank-top',type=int,default=30)
    pa.add_argument('--save-every',type=int,default=100)
    a=pa.parse_args()
    start=time.time()
    basedir=Path(a.project).resolve();sys.path.insert(0,str(basedir))
    from app.store import tokenize
    db_path=Path(a.db_path).resolve() if a.db_path else basedir/'data/knowledge.db'
    db=get_db(db_path)
    docs=chunks_and_docs(db)
    items=load_items(a.dataset,a.max_queries)
    if not items:raise RuntimeError('No queries: expected dataset rows with query field')
    if a.sample_queries and a.sample_queries<len(items):
        import random
        rng=random.Random(a.seed)
        items=sorted(rng.sample(items, a.sample_queries),key=lambda x:x[0])
        print(f'Sampled {len(items)} questions with seed={a.seed}',flush=True)
    print(f'loaded {len(items)} queries, {len(docs)} documents; mode={a.mode}; db={db_path}',flush=True)
    # Exact gold document indexability should be checked separately from Top-10 retrieval.
    url_index={d['url_key'] for d in docs.values() if d['url_key']}
    title_index={d['title_key'] for d in docs.values() if d['title_key']}
    title_no_url={d['title_key'] for d in docs.values() if d['title_key'] and not d['url_key']}
    chunk_to_doc={int(r['id']):int(r['document_id']) for r in db.execute('SELECT id,document_id FROM chunks')}
    results=[]
    # Baseline and hybrid use identical FTS5 BM25 queries.
    fts=[fts_chunks(db,item.get('query',''),tokenize,100) for _,item in items]
    print('SQLite FTS5 complete',flush=True)
    id_order=None;dense_ranks=None;chunk_body=None
    if a.mode!='baseline':
        chunks=[(int(r['id']),str(r['body'])) for r in db.execute('SELECT id,body FROM chunks ORDER BY id')]
        id_order=[ch[0] for ch in chunks]
        from numpy import save,load
        ensure_dir(a.out)
        vectors_file=Path(a.out)/'corpus_embeddings.npy'
        meta_file=Path(a.out)/'corpus_embeddings.meta.json'
        sha=hashlib.sha256()
        for cid, body in chunks:
            sha.update(str(cid).encode('ascii'));sha.update(b'\0');sha.update(body.encode('utf-8'));sha.update(b'\0')
        metadata={'model':a.embedding_model,'max_seq_length':a.max_seq_length,'chunks_sha256':sha.hexdigest(),'chunk_count':len(chunks)}
        if vectors_file.exists() or meta_file.exists():
            if not vectors_file.exists() or not meta_file.exists():raise RuntimeError('Incomplete embedding cache. Use a new --out directory.')
            saved=json.loads(meta_file.read_text(encoding='utf-8'))
            if saved != metadata:raise RuntimeError('Embedding cache mismatch (model/config/corpus); use a different --out')
            vec=load(vectors_file,mmap_mode='r')
            if vec.shape[0]!=len(chunks):raise RuntimeError('Corrupt embedding cache')
            print('Verified embedding cache loaded',flush=True)
            from sentence_transformers import SentenceTransformer
            import torch
            import torch_musa
            if not torch.musa.is_available():raise RuntimeError('No MUSA GPU for query embeddings')
            model=SentenceTransformer(a.embedding_model,device='musa',trust_remote_code=True)
            model.max_seq_length=a.max_seq_length
            qemb=model.encode([x['query'] for _,x in items],batch_size=a.batch_size,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
            del model
            torch.musa.empty_cache()
        else:
            vec,qemb=encode_hybrid(a.embedding_model,chunks,[x['query'] for _,x in items],a.max_seq_length,a.batch_size)
            temp_vec=str(vectors_file)+'.tmp'
            with open(temp_vec,'wb') as writer:save(writer,vec)
            os.replace(temp_vec,vectors_file)
            temp_meta=str(meta_file)+'.tmp'
            Path(temp_meta).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(temp_meta,meta_file)
        dense_ranks=[[id_order[j] for j in order] for order in dense_topk(vec,qemb,100)]
        print('Dense ranking complete',flush=True)
        if a.mode=='hybrid_bge':
            from sentence_transformers import CrossEncoder
            import torch
            import torch_musa
            if not torch.musa.is_available():raise RuntimeError('No MUSA for reranker')
            ranker=CrossEncoder(a.rerank_model,device='musa',max_length=a.max_seq_length,trust_remote_code=True)
            chunk_body=dict(chunks)
    # Store compact ranked chunk IDs for reproducible, cheap CPU fusion ablations.
    ranks_output = Path(a.out)/'retrieval_ranks.jsonl'
    if a.mode=='hybrid':
        rank_stream=ranks_output.open('w', encoding='utf8')
    else:
        rank_stream=None
    # Persist cross-encoder raw scores for K=10/30/60 comparisons without rescoring.
    rerank_stream=None
    if a.mode=='hybrid_bge':
        rerank_stream=(Path(a.out)/'rerank_scores.jsonl').open('w', encoding='utf8')
    for pos,(idx,row) in enumerate(items):
        gold=gold_keys(row)
        indexable=sum(int((u in url_index or (t in title_no_url)) if u else (t in title_index)) for u,t in gold)
        row_out={'index':idx,'question_type':row.get('question_type',''), 'question':row['query'],'answer':row.get('answer',''), 'gold_total':len(gold),'gold_indexable_count':indexable}
        def score(method,ranks):
            candidates=unique_docs(ranks,chunk_to_doc,10)
            vals=metric(candidates,docs,gold)
            for field in ('covered','all','hit'):
                row_out[method+'_'+field]=vals[field]
            row_out[method+'_document_ids']='|'.join(str(x) for x in candidates)
            row_out[method+'_document_titles']=' | '.join(docs[x]['title'] for x in candidates)
        score('bm25',fts[pos])
        if dense_ranks is not None:
            score('dense',dense_ranks[pos])
            fused=rrf(fts[pos],dense_ranks[pos])
            score('hybrid',fused)
            if rank_stream is not None:
                rank_stream.write(json.dumps({'index':idx,'question_type':row.get('question_type',''),
                    'bm25':fts[pos], 'dense':dense_ranks[pos], 'hybrid':fused[:120]}, ensure_ascii=False)+'\n')
                if (pos+1)%100==0: rank_stream.flush()
            if a.mode=='hybrid_bge':
                subset=fused[:max(10,a.rerank_top)]
                # Need original scoring order; model.predict supplies one score per pair.
                pairs=[(row['query'],chunk_body[ch]) for ch in subset]
                values=ranker.predict(pairs,batch_size=a.batch_size,show_progress_bar=False)
                if rerank_stream is not None:
                    rerank_stream.write(json.dumps({'index':idx, 'question_type':row.get('question_type',''),
                        'chunk_ids':subset, 'scores':[float(v) for v in values]},ensure_ascii=False)+'\n')
                    if (pos+1)%20==0: rerank_stream.flush()
                ordered=sorted(range(len(subset)),key=lambda j:(-float(values[j]),j))
                score('hybrid_bge',[subset[j] for j in ordered])
        results.append(row_out)
        if (pos+1)%a.save_every==0:
            print(f'{pos+1}/{len(items)} queries; elapsed={time.time()-start:.1f}s',flush=True)
            dump(results,summarize(results),a.out,{'mode':a.mode,'partial':True,'n_questions':len(items)})
    if rank_stream: rank_stream.close()
    if rerank_stream: rerank_stream.close()
    dump(results,summarize(results),a.out,{'mode':a.mode,'partial':False,'n_questions':len(items),'elapsed_seconds':round(time.time()-start,2), 'db':str(db_path), 'embed_model':a.embedding_model if a.mode!='baseline' else None})
    (Path(a.out)/'DONE').write_text('completed\n')
    print((Path(a.out)/'summary.txt').read_text(),flush=True)
    print(f'DONE {a.out}',flush=True)

if __name__=='__main__':main()
