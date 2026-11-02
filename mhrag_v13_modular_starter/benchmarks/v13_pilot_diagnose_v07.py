#!/usr/bin/env python3
"""Read-only diagnosis of v0.6/v0.7 development-pilot results. No GPU calls."""
import argparse
from collections import Counter
import json
from pathlib import Path


def diagnose(path):
    rows=[]
    with Path(path).open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if not line.strip():continue
            try: row=json.loads(line)
            except json.JSONDecodeError as exc:raise ValueError(f'Invalid JSON at line {n}') from exc
            if row.get('arm')!='G':continue
            result=row.get('result') or {}
            identity=row.get('identity') or result.get('identity') or {}
            routing=row.get('routing') or result.get('routing') or {}
            spans=(result.get('evidence') or {}).get('local_spans') or []
            status=row.get('generation_status') or result.get('generation_status') or 'unknown'
            if routing.get('mode'):
                reason=routing.get('reason') or 'none'
                stage=routing['mode']
            elif not identity.get('complete'):
                stage='legacy_identity_block'
                reason='unresolved_names' if identity.get('unresolved') else 'no_confident_study_identity'
            elif not spans:
                stage='legacy_no_local_sections';reason='zero_local_spans'
            else:
                stage='legacy_generation';reason='inspect_generation_diagnostics'
            rows.append({'case_id':row.get('case_id'), 'generation_status':status,
                         'routing_stage':stage, 'routing_reason':reason,
                         'identity_complete':bool(identity.get('complete')),
                         'unresolved':identity.get('unresolved') or [],
                         'local_spans':len(spans),
                         'accepted':int(row.get('accepted_claim_count') or 0)})
    return {'cases':len(rows),'route_counts':dict(Counter(x['routing_stage'] for x in rows)),
            'generation_counts':dict(Counter(x['generation_status'] for x in rows)),
            'rows':rows,
            'warning':'No medical accuracy or Gold coverage is inferred.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('results_jsonl')
    p.add_argument('--json',action='store_true')
    args=p.parse_args()
    r=diagnose(args.results_jsonl)
    if args.json:
        print(json.dumps(r,ensure_ascii=False,indent=2))
    else:
        print('G tasks:',r['cases'])
        print('Routes:',r['route_counts'])
        print('Generation:',r['generation_counts'])
        for x in r['rows']:
            print(f"{x['case_id']:<34} {x['routing_stage']:<27} {x['generation_status']:<9} "
                  f"spans={x['local_spans']} accepted={x['accepted']} reason={x['routing_reason']}")
        print('Medical accuracy: NOT EVALUATED')

if __name__=='__main__':main()
