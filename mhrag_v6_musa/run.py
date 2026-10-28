#!/usr/bin/env python3
"""Independent GPU-2 focused experiment: 750-char BM25 vs Hybrid+CE, FULL 2255 labeled questions.
Do not write or reuse v5 output dirs. All work under --root.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

P = Path(__file__).resolve().parent.parent
BGE = Path('/root/autodl-tmp/models/bge-m3')
HUB = Path('/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots')
DATA = P/'dataset/MultiHopRAG.json'
DB = P/'mhrag_chat_mvp/data/knowledge.db'
BASE = P/'results/overnight_v4_guarded_20261027_050407'
BASERANK = BASE/'hybrid_8k/retrieval_ranks.jsonl'
BASESCORE = BASE/'ce_hybrid/rerank_scores.jsonl'


def ts():return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def pick_model():
    if not HUB.is_dir():return None
    for d in sorted(HUB.iterdir()):
        if (d/'config.json').is_file() and ((d/'model.safetensors').is_file() or (d/'pytorch_model.bin').is_file()):
            return d
    return None


def ensure(require_gpu=True):
    paths=[DATA,DB,BGE/'modules.json',BASERANK,BASESCORE,BASE/'ce_hybrid/DONE',
           P/'mhrag_overnight_v5/stress.py',P/'mhrag_overnight_v5/evaluate.py',
           P/'mhrag_v6_musa/hybrid_musa.py',
           P/'mhrag_v6_musa/rerank_musa.py']
    bad=[str(x) for x in paths if not x.is_file()]
    model=pick_model()
    if not model:bad.append('complete reranker snapshot at '+str(HUB))
    if bad:raise RuntimeError('Missing files:\n  '+'\n  '.join(bad))
    with sqlite3.connect('file:'+str(DB)+'?mode=ro',uri=True) as cn:
        docs=cn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        chunks=cn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    if require_gpu:
        import torch
        import torch_musa
        if not torch.musa.is_available():raise RuntimeError('MUSA unavailable')
        print('GPU:',torch.musa.get_device_name(0),flush=True)
    print('Preflight OK; documents=',docs,'chunks=',chunks,'model=',model,flush=True)
    return model


def report(root):
    rows=[]
    for name in ('bm25_750','hybrid_750'):
        path=root/('eval_'+name)/'summary.json'
        if not path.exists():continue
        a=json.loads(path.read_text(encoding='utf8'))
        m=a.get('holdout_by_method',{})
        d=a.get('paired_vs_baseline',{})
        rec={'name':name,'n_candidate':a.get('n_candidate'),'n_baseline':a.get('n_baseline'),
             'n_paired_holdout':a.get('n_paired_holdout'),
             'coverage':m.get('ce100_coverage',{}),'all_gold':m.get('ce100_all',{}),
             'paired_delta_coverage_vs_original_CE':d.get('ce100_coverage',{}),
             'paired_delta_all_gold_vs_original_CE':d.get('ce100_all',{})}
        rows.append(rec)
    (root/'V6_COMPARISON.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf8')
    for r in rows:
        print('RESULT',r['name'],'paired_n',r['n_paired_holdout'],
              'CE Coverage',r['coverage'].get('mean'),'CE AllGold',r['all_gold'].get('mean'),
              'Delta AllGold vs original CE',r['paired_delta_all_gold_vs_original_CE'].get('delta'),flush=True)
    print('Results at',root/'V6_COMPARISON.json',flush=True)


def run(args):
    os.chdir(P)
    os.environ.update({'HF_HOME':'/root/autodl-tmp/hf_cache','HF_HUB_OFFLINE':'1',
                       'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    model=ensure(require_gpu=not args.dry_run)
    root=Path(args.root).resolve()
    if str(root)==str(P) or P not in root.parents:raise RuntimeError('--root must be under project, not project root')
    root.mkdir(parents=True,exist_ok=True)
    disk=shutil.disk_usage(str(root))
    if disk.free < 8*1024**3:raise RuntimeError('Less than 8 GB free disk')
    (root/'config.json').write_text(json.dumps({'length':512,'all_labeled_queries':True,
      'backend':'musa','original_articles_not_independent_external_data':True,'created_at':ts(),
      'baseline_rank_sha256':hashlib.sha256(BASERANK.read_bytes()).hexdigest(),
      'baseline_score_sha256':hashlib.sha256(BASESCORE.read_bytes()).hexdigest()},indent=2))
    corpus=root/'rechunk750'
    bge=corpus/'hybrid_bge'
    commands=[
     ('rechunk750',corpus/'DONE',[
       sys.executable,'-u','mhrag_overnight_v5/stress.py','--window','750','--overlap','150',
       '--db',str(DB),'--dataset',str(DATA),'--out',str(corpus),'--questions','0']),
     ('hybrid_bge',bge/'DONE',[
       sys.executable,'-u','mhrag_v6_musa/hybrid_musa.py',
       '--mode','hybrid','--db-path',str(corpus/'corpus.db'),'--out',str(bge),
       '--embedding-model',str(BGE),'--batch-size','4','--max-seq-length','512',
       '--dataset',str(DATA)]),
    ]
    for name,rank in [('bm25_750',corpus/'retrieval_ranks.jsonl'),('hybrid_750',bge/'retrieval_ranks.jsonl')]:
        ce=root/('ce_'+name)
        evalpath=root/('eval_'+name)
        commands.append(('ce_'+name,ce/'DONE',[
          sys.executable,'-u','mhrag_v6_musa/rerank_musa.py',
          '--ranks',str(rank),'--db-path',str(corpus/'corpus.db'),'--dataset',str(DATA),
          '--pool','hybrid','--sample','0','--top','100','--max-length','512',
          '--batch-size','4','--model',str(model),'--out',str(ce)]))
        commands.append(('eval_'+name,evalpath/'DONE',[
          sys.executable,'-u','mhrag_overnight_v5/evaluate.py',
          '--baseline',str(BASESCORE),'--baseline-ranks',str(BASERANK),
          '--candidate',str(rank),'--scores',str(ce/'rerank_scores.jsonl'),
          '--candidate-db',str(corpus/'corpus.db'),'--db',str(DB),'--dataset',str(DATA),
          '--out',str(evalpath),'--questions','0']))
    if args.dry_run:
        for name,marker,cmd in commands:
            print(name,'=>',' '.join(map(str,cmd)))
        return
    (root/'STATUS').write_text('RUNNING\n')
    try:
        for name,marker,command in commands:
            if marker.is_file():
                print('SKIP',name,'already DONE',flush=True);continue
            print('START',ts(),name,flush=True)
            t=time.monotonic()
            logpath=root/(name+'.log')
            with logpath.open('a',encoding='utf8') as stream:
                stream.write('\n'+ts()+' COMMAND '+subprocess.list2cmdline(list(map(str,command)))+'\n')
                stream.flush()
                ret=subprocess.call(command,cwd=P,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
            if ret or not marker.is_file():
                raise RuntimeError(f'{name} failed ret={ret}; inspect {logpath}')
            print('END',ts(),name,round(time.monotonic()-t,1),'seconds',flush=True)
        report(root)
        (root/'STATUS').write_text('DONE\n')
        print('ALL DONE',flush=True)
    except BaseException as e:
        (root/'STATUS').write_text('FAILED: '+repr(e)+'\n')
        raise

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='results/v6_focus_750_musa')
    a.add_argument('--dry-run',action='store_true')
    args=a.parse_args()
    run(args)
