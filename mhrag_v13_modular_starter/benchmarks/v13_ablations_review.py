#!/usr/bin/env python3
"""Prepare independent reviewer sheets for complete DEV F/E1/E2/E3 ablation.

Reviewer sheets omit arm identity, system diagnostics and clinical truth labels.
Private admin arm key is separate; answers can still reveal stylistic identity.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import random
import secrets
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.v13_ablations_runner import ARMS


def export_reviews(outdir: Path, *, output: Path | None = None):
    outdir=Path(outdir).resolve()
    reviewdir=Path(output).resolve() if output is not None else outdir.parent/(outdir.name+'_review')
    if reviewdir.exists():raise ValueError('Refusing to overwrite existing blinded reviewer material')
    manifest=json.loads((outdir/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((outdir/'summary.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='historical_dev_91_not_holdout_NOT_MEDICALLY_VALIDATED':
        raise ValueError('This export only supports the historical 91-case DEV ablation')
    if summary.get('state')!='complete' or summary.get('total_runtime_errors')!=0 or summary.get('completed')!=364:
        raise ValueError('Only complete error-free 364-job DEV experiments can be exported')
    cases=json.loads(Path(manifest['config']['cases']).read_text(encoding='utf-8'))['cases']
    lookup={c['case_id']:c['question'] for c in cases}
    records=[json.loads(line) for line in (outdir/'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    keyed={(r['case_id'],r['arm']):r for r in records}
    if len(keyed)!=364 or len(records)!=364:raise ValueError('Incomplete or duplicate case/arm records')
    seed=secrets.randbits(64)
    rng=random.Random(seed)
    rows=[]; admin=[]
    for case in cases:
        cid=case['case_id']
        ordered=list(ARMS);rng.shuffle(ordered)
        for i,arm in enumerate(ordered,1):
            row=keyed[(cid,arm)]
            if row['status']!='ok':raise ValueError('Error in source record')
            tag=f'{cid}:R{i}'
            rows.append({'case_id':cid,'blind_id':tag,'question':lookup[cid],
                         'answer':row.get('answer',''),
                         'task_answered_score_0_to_2':'',
                         'medical_factuality_from_pdf_correct_incorrect_uncertain':'',
                         'citation_supported_yes_no_uncertain':'',
                         'abstention_appropriate_yes_no_not_applicable':'',
                         'critical_safety_error_yes_no':'',
                         'pdf_pages_verified':'',
                         'reviewer_notes':''})
            admin.append({'case_id':cid,'blind_id':tag,'arm':arm})
    reviewdir.mkdir(parents=True)
    columns=list(rows[0])
    for reviewer in ('R1','R2'):
        with (reviewdir/f'blinded_{reviewer}_364_answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns)
            w.writeheader();w.writerows(rows)
    with (reviewdir/'ADMIN_UNBLIND_AFTER_ADJUDICATION.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(admin[0]));w.writeheader();w.writerows(admin)
    (reviewdir/'README_REVIEW.txt').write_text(
        '91题历史开发集，每题4条匿名答案，共364行。R1/R2独立评判并记录原PDF页码，'
        '分歧裁决完成前不要查看 ADMIN_UNBLIND_AFTER_ADJUDICATION.csv。'
        '答案文本的语气和引文格式仍可能暴露方法，因此并非完全双盲。'
        '请勿以自动引用/数字校验替代医学事实或临床正确性。\n',encoding='utf-8')
    return reviewdir,len(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--out')
    a=p.parse_args()
    folder,count=export_reviews(Path(a.run),output=Path(a.out) if a.out else None)
    print('BLIND_DEV_REVIEW_EXPORT_OK',folder,'answers=',count)

if __name__=='__main__':main()
