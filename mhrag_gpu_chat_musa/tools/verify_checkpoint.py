#!/usr/bin/env python3
"""Check v6 JSONL append checkpoints. --repair-tail backs up and trims a broken final row ONLY."""
from __future__ import annotations
import argparse,json,shutil,time
from pathlib import Path

def check(path,repair=False):
    if not path.exists():return print('MISSING',path)
    data=path.read_bytes()
    lines=data.splitlines(keepends=True)
    bad=[];offset=0;valid=0
    for i,line in enumerate(lines):
        try:
            if not line.endswith(b'\n'):raise ValueError('missing newline')
            row=json.loads(line)
            if not (isinstance(row,dict) and len(row['chunk_ids'])==len(row['scores'])):raise ValueError('invalid scores row')
            valid+=1
        except Exception as e:bad.append((i,offset,str(e)))
        offset+=len(line)
    print(path,'valid_records=',valid,'bad_records=',len(bad))
    if bad:print('BAD:',bad[:5])
    if repair and bad:
        if len(bad)!=1 or bad[0][0]!=len(lines)-1:raise RuntimeError('Bad middle row; refuse automatic repair')
        backup=path.with_name(path.name+'.backup_'+str(int(time.time())))
        shutil.copy2(path,backup)
        with path.open('r+b') as f:f.truncate(bad[0][1])
        print('REPAIRED trailing broken record. Backup:',backup)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='results/v6_focus_750_musa');a.add_argument('--repair-tail',action='store_true');args=a.parse_args()
    for stage in ('ce_bm25_750','ce_hybrid_750'):
        check(Path(args.root)/stage/'rerank_scores.jsonl',args.repair_tail)
