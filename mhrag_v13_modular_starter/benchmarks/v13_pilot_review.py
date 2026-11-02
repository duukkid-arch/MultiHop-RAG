#!/usr/bin/env python3
"""Export an arm-masked DEV answer adjudication worksheet. Never auto-award clinical correctness."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def main():
    p = argparse.ArgumentParser(description='Create blinded-ish clinical review CSV from completed DEV paired pilot')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True,
                   help='Directory for output; must NOT exist')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    run = args.run.resolve()
    if args.out.exists():
        p.error('Output exists; refusing to overwrite')
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    plan = manifest['plan']
    if len(rows) != len(plan) or any((r.get('case_id'), r.get('arm')) != (job['case_id'], job['arm']) for r, job in zip(rows, plan)):
        p.error('Cannot review incomplete or corrupted pilot')
    if any(r.get('status') != 'ok' for r in rows):
        p.error('Cannot review pilot containing runtime errors')
    by_case = {}
    for r in rows:
        by_case.setdefault(r['case_id'], {})[r['arm']] = r
    if any(set(group) != {'F', 'G'} for group in by_case.values()):
        p.error('Missing paired arm')
    rng = random.Random(args.seed)
    output = []
    key = []
    for case_id, group in by_case.items():
        flip = rng.choice((True, False))
        for arm, label in (('F', 'B' if flip else 'A'), ('G', 'A' if flip else 'B')):
            r = group[arm]
            output.append({'case_id': case_id, 'response_id': label,
                           'answer': r.get('answer', ''),
                           'reported_document_ids': ','.join(map(str, r.get('api_doc_ids', []))),
                           'reviewer_id': '', 'population_intervention_correct_0_1': '',
                           'primary_outcome_correct_0_1': '',
                           'results_correct_0_1': '',
                           'citations_support_claims_0_1': '',
                           'abstention_appropriate_0_1': '',
                           'fully_grounded_success_0_1': '',
                           'comments': ''})
            key.append({'case_id': case_id, 'response_id': label, 'arm': arm})
    rng.shuffle(output)
    args.out.mkdir(parents=True)
    csv_path = args.out / 'for_reviewers_only.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader();writer.writerows(output)
    (args.out / 'private_unblinding_key.json').write_text(json.dumps(key, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.out / 'README.txt').write_text(
        'DEV-only pilot. Send ONLY for_reviewers_only.csv to reviewers. Keep private_unblinding_key.json hidden.\n'
        'A/B labels are randomly assigned separately for each question; response wording may still reveal the system.\n'
        'Have two reviewers fill independent copies, resolve disagreements, and evaluate against original PDF.\n'
        'Automatic accepted_claim_count is NOT clinical correctness and is not an outcome metric.\n', encoding='utf-8')
    print('REVIEW_EXPORT_OK', str(csv_path), 'rows=', len(output))

if __name__ == '__main__':
    main()
