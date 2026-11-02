#!/usr/bin/env python3
"""MedHop original: CPU-friendly, per-example TF-IDF retrieval + bounded two-hop candidate ranking.
No gold answers are used in prediction. This is a transparent agentic retrieval baseline, not LLM generation.
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DEFAULT = '/root/autodl-tmp/medhop_benchmark_private/medhop_original'
TOKEN = re.compile(r'(?u)\b[\w-]+\b')

def norm(s):
    return ' '.join(TOKEN.findall(str(s).lower().replace('_', ' ')))

def load_data(root, split):
    root = Path(root)
    if root.is_file():
        return parse_json(root)
    # HuggingFace save_to_disk format
    if (root / 'dataset_dict.json').exists() or (root / 'state.json').exists():
        from datasets import load_from_disk
        ds = load_from_disk(str(root))
        if hasattr(ds, 'keys'):
            ds = ds[split if split in ds else ('dev' if split == 'validation' and 'dev' in ds else split)]
        return [dict(x) for x in ds]
    names = ['validation.json','dev.json','valid.json'] if split == 'validation' else ['train.json']
    paths = [root / x for x in names]
    paths += [p for p in root.rglob('*.json') if p.name in names]
    for p in paths:
        if p.exists():
            return parse_json(p)
    # Arrow cache in split subfolder
    for p in [root / split, root / ('dev' if split == 'validation' else split)]:
        if (p / 'state.json').exists():
            from datasets import load_from_disk
            return [dict(x) for x in load_from_disk(str(p))]
    raise FileNotFoundError(f'No {split} data found under {root}. Inspect: find {root} -maxdepth 3 -type f | head')

def parse_json(path):
    obj = json.loads(path.read_text(encoding='utf8'))
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
    raise ValueError(f'Unsupported JSON layout: {path}')

def validate(examples):
    for i, x in enumerate(examples):
        missing = {'query','supports','candidates'} - set(x)
        if missing: raise ValueError(f'Row {i} missing fields: {missing}; found {list(x)}')
        if not isinstance(x['supports'], list) or not isinstance(x['candidates'], list):
            raise ValueError(f'Row {i}: supports/candidates must be lists')

def support_score(text, candidate):
    # Require full candidate phrase boundary to avoid substring false positives.
    phrase = norm(candidate)
    hay = norm(text)
    return bool(re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', hay)) if phrase else False

def predict(row, mode='agentic', topk=5):
    docs = [str(x) for x in row['supports'] if str(x).strip()]
    cand = list(row['candidates'])
    query = str(row['query'])
    if not cand: return {'prediction':'', 'trace':[], 'retrieved':[]}
    if not docs: return {'prediction':cand[0], 'trace':[], 'retrieved':[]}
    # Fit only on this example's support documents: respects the original task context.
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), sublinear_tf=True)
    try:
        mat = vec.fit_transform(docs + [query] + cand)
    except ValueError:  # all stop words
        return {'prediction':cand[0], 'trace':[], 'retrieved':[]}
    D = mat[:len(docs)]
    q = mat[len(docs)]
    C = mat[len(docs)+1:]
    qsim = cosine_similarity(D, q).ravel()
    csim = cosine_similarity(D, C)
    first = [int(j) for j in np.argsort(-qsim, kind='stable')[:max(1, topk)]]
    scores = np.zeros(len(cand), dtype=float)
    trace = [{'hop':1, 'query':query, 'doc_indices':[int(j) for j in first]}]
    if mode == 'single':
        pool = first
        for i, c in enumerate(cand):
            vals = [((1.0 if support_score(docs[j], c) else 0.0) + 0.15*csim[j,i]) * (0.1+qsim[j]) for j in pool]
            scores[i] = max(vals, default=0)
    else:
        # Action: retrieve bridge documents mentioning the starting entity, but not the candidate answer.
        # Original MedHop queries are often `interacts_with drug_name`.
        parts = query.split(' ', 1)
        seed = parts[-1] if len(parts)>1 else query
        seed = seed.replace('_', ' ')
        bridge = [j for j in first if support_score(docs[j],seed)]
        if not bridge: bridge = first[:2]
        second = set(first)
        for j in bridge[:3]:
            # re-retrieval using first-hop document as a weak query expansion
            near = cosine_similarity(D, D[j]).ravel()
            for ix in np.argsort(-near, kind='stable')[:topk]: second.add(int(ix))
        pool = sorted(second)
        trace.append({'hop':2,'seed':seed,'bridge_doc_indices':[int(i) for i in bridge[:3]],'doc_indices':pool})
        for i,c in enumerate(cand):
            # Two distinct support passages: seed bridge and candidate hit.
            hit = [j for j in pool if support_score(docs[j],c)]
            if not hit: continue
            best = 0.0
            for a in bridge[:3]:
                for b in hit:
                    if a == b: continue
                    joint = cosine_similarity(D[a],D[b])[0,0]
                    val = 1 + 0.2*qsim[a] + 0.2*qsim[b] + 0.4*joint + 0.1*csim[b,i]
                    best = max(best,val)
            if best == 0: best = 0.1 + max(qsim[j] for j in hit)
            scores[i] = best
    winner = int(np.argmax(scores))
    return {'prediction':str(cand[winner]), 'trace':trace,
            'retrieved':pool, 'confidence_proxy':float(scores[winner])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',default=ROOT_DEFAULT)
    ap.add_argument('--split',choices=['train','validation'],default='validation')
    ap.add_argument('--mode', choices=['single','agentic'], default='agentic')
    ap.add_argument('--limit',type=int,default=0,help='0 = entire split')
    ap.add_argument('--topk',type=int,default=5)
    ap.add_argument('--out',default='results/medhop_agentic.jsonl')
    args = ap.parse_args()
    data = load_data(args.data,args.split)
    validate(data)
    if args.limit: data = data[:args.limit]
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    correct=total=0; start=time.time()
    with out.open('w',encoding='utf8') as f:
        for idx,row in enumerate(data):
            res = predict(row,args.mode,args.topk)
            gold = row.get('answer')
            ok = norm(res['prediction']) == norm(gold) if gold is not None else None
            if ok is not None: correct+=int(ok);total+=1
            record = {'id':str(row.get('id',idx)), 'query':row['query'], 'prediction':res['prediction'],
                      'gold':gold, 'correct':ok, 'trace':res['trace'],
                      'retrieved':res['retrieved'], 'confidence_proxy':res.get('confidence_proxy')}
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    summary={'split':args.split,'mode':args.mode,'examples':len(data), 'scored':total,
             'accuracy':round(correct/total,4) if total else None,
             'seconds':round(time.time()-start,2),'output':str(out)}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    (out.parent / (out.stem+'_metrics.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
if __name__=='__main__': main()
