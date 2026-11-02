#!/usr/bin/env python3
"""v13 DEV-only paired F vs G pilot with strict fingerprints, lock & resume.

F is the *existing* standard RAG chat API (agentic=False).
G is the currently installed v13 grounded sidecar, NOT an independently validated E3.
No gold/reference is loaded into the runtime, and clinical correctness is NOT scored.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            data = f.read(4 * 1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def source_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    paths = sorted((root / 'mhrag_v13').glob('*.py'))
    paths += [root / 'benchmarks' / 'v13_pilot_runner.py']
    for file in paths:
        h.update(file.relative_to(root).as_posix().encode('utf-8'))
        h.update(bytes.fromhex(sha256_file(file)))
    return h.hexdigest()


def load_cases(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or raw.get('dataset_kind') != 'development_only_no_gold_for_runtime':
        raise ValueError('Only explicit DEV no-gold pilot input is allowed')
    cases = raw.get('cases')
    if not isinstance(cases, list) or len(cases) != 16:
        raise ValueError('Pilot must have exactly 16 DEV questions')
    seen = set()
    for c in cases:
        if not isinstance(c, dict) or set(c) != {'case_id', 'question', 'development_phase'}:
            raise ValueError('DEV runtime case must contain only id, question, phase; no gold or references')
        if not isinstance(c['case_id'], str) or not c['case_id'].strip() or c['case_id'] in seen:
            raise ValueError('Invalid or duplicate case_id')
        if not isinstance(c['question'], str) or len(c['question'].strip()) < 12:
            raise ValueError('Missing/question too short')
        if c['development_phase'] not in ('baseline', 'stress', 'catalog'):
            raise ValueError('Unrecognized DEV phase')
        seen.add(c['case_id'])
    return cases


def make_plan(cases: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    for c in cases:
        arms = ['F', 'G']
        rng.shuffle(arms)
        for arm in arms:
            plan.append({'case_id': c['case_id'], 'arm': arm})
    return plan


def kb_info(db_path: Path) -> dict:
    with sqlite3.connect(f'file:{db_path.as_posix()}?mode=ro', uri=True) as conn:
        docs = conn.execute('SELECT count(*) FROM documents').fetchone()[0]
        chunks = conn.execute('SELECT count(*) FROM chunks').fetchone()[0]
        return {'documents': docs, 'chunks': chunks}


def local_model_ok(qwen_url: str, model: str) -> None:
    req = urllib.request.Request(qwen_url.rstrip('/') + '/models')
    with urllib.request.urlopen(req, timeout=8) as res:
        payload = json.load(res)
    names = {x.get('id') for x in payload.get('data', []) if isinstance(x, dict)}
    if model not in names:
        raise RuntimeError(f'Local Qwen model {model!r} not present at {qwen_url!r}')


def make_manifest(args, cases: list[dict], plan: list[dict], info: dict) -> dict:
    return {
        'protocol': 'v13-pilot-paired-v0.7-routing',
        'kind': 'development_diagnostic_not_medical_accuracy_not_holdout',
        'arms': {'F': 'existing RAG agentic=false', 'G': 'current v13 grounded sidecar; NOT validated E3'},
        'run_config': {
            'db_path': str(Path(args.db).resolve()),
            'cases_path': str(Path(args.cases).resolve()),
            'aliases_path': str(Path(args.aliases).resolve()) if args.aliases else None,
            'chat_url': args.chat_url,
            'qwen_url': args.qwen_url,
            'model': args.model,
            'mode': args.mode,
            'judge': args.judge,
            'draft_strategy': args.draft_strategy,
            'max_draft_calls': args.max_draft_calls,
            'max_http': args.max_http,
            'seed': args.seed,
        },
        'input_sha256': {
            'db': sha256_file(Path(args.db).resolve()),
            'cases': sha256_file(Path(args.cases).resolve()),
            'aliases': sha256_file(Path(args.aliases).resolve()) if args.aliases else None,
            'source_tree': source_fingerprint(ROOT),
        },
        'kb': info,
        'plan': plan,
        'labels_accessible_to_runtime': False,
    }


def read_existing(path: Path, plan: list[dict]) -> list[dict]:
    completed = []
    if not path.exists():
        return completed
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Corrupt results.jsonl at line {line_no}; manually inspect before resume') from exc
            if not isinstance(row, dict) or line_no > len(plan) or (row.get('case_id'), row.get('arm')) != (plan[line_no - 1]['case_id'], plan[line_no - 1]['arm']):
                raise ValueError(f'Unexpected/duplicate/out-of-order result at line {line_no}')
            if row.get('status') not in ('ok', 'error'):
                raise ValueError(f'Invalid status at line {line_no}')
            completed.append(row)
    return completed


def run_job(job: dict, questions: dict, retriever, controller, mode: str) -> dict:
    tic = time.monotonic()
    q = questions[job['case_id']]
    record = {'case_id': job['case_id'], 'arm': job['arm'], 'status': 'ok',
              'utc': datetime.now(timezone.utc).isoformat()}
    try:
        if job['arm'] == 'F':
            output = retriever.search(q, agentic=False)
            record.update({'answer': output.answer,
                           'api_doc_ids': sorted({x.document_id for x in output.sources}),
                           'api_source_count': len(output.sources),
                           'http_calls': 1,
                           'accepted_claim_count': None,
                           'medical_claims_verified': False})
        else:
            result = controller.run(q, evidence_only=(mode == 'evidence'))
            accepted = result.get('claims', {}).get('accepted', [])
            record.update({
                'answer': result['answer'],
                'api_doc_ids': result['evidence']['api_doc_ids'],
                'local_doc_ids': result['evidence']['local_doc_ids'],
                'http_calls': result['retrieval']['http_calls'],
                'identity': result.get('identity'),
                'routing': result.get('routing'),
                'slots': result['slots'],
                'decision': result['decision'],
                'generation_status': result.get('generation_status'),
                'generation_error': result.get('generation_error'),
                'generation_calls': result.get('generation_calls'),
                'generation_diagnostics': result.get('generation_diagnostics', []),
                'draft_claim_count': result.get('draft_claim_count'),
                'accepted_claim_count': len(accepted),
                'accepted_study_ids': sorted({a['claim']['study_id'] for a in accepted}),
                'rejected_claim_count': result.get('rejected_claim_count'),
                'rejected_checks': [str(r.get('check')) for r in result.get('claims', {}).get('rejected', [])],
                'medical_claims_verified': False,
                'human_medical_review_required': True,
                'result': result,
            })
    except Exception as exc:
        record.update({'status': 'error', 'error_type': type(exc).__name__, 'error_message': str(exc)[:240]})
    record['wall_ms'] = round((time.monotonic() - tic) * 1000, 1)
    return record


def summary(rows: list[dict], plan: list[dict], deadline_hit=False, aborted=False, mode='qwen') -> dict:
    keyed = {(r['case_id'], r['arm']): r for r in rows}
    arm_stats = {}
    for arm in ('F', 'G'):
        group = [r for r in rows if r['arm'] == arm]
        ok = [r for r in group if r['status'] == 'ok']
        arm_stats[arm] = {'completed': len(group), 'transport_or_runtime_errors': len(group)-len(ok),
                           'median_ms': round(statistics.median(r['wall_ms'] for r in ok), 1) if ok else None}
        if arm == 'G':
            arm_stats[arm].update({
                'cases_with_accepted_claim': sum(int((r.get('accepted_claim_count') or 0) > 0) for r in ok),
                'total_accepted_claims': sum(r.get('accepted_claim_count') or 0 for r in ok),
                'cases_with_claims_from_two_studies': sum(len(r.get('accepted_study_ids', []))>=2 for r in ok),
                'generation_statuses': dict(Counter(r.get('generation_status', 'unknown') for r in ok)),
                'generation_errors': dict(Counter(str(r['generation_error']) for r in ok if r.get('generation_error'))),
                'routing_modes': dict(Counter((r.get('routing') or {}).get('mode','unknown') for r in ok)),
                'routing_reasons': dict(Counter((r.get('routing') or {}).get('reason','unknown') for r in ok)),
                'abstention_count': sum(r.get('decision', {}).get('decision') != 'answer' for r in ok),
            })
    paired = sum(('F' in {x[1] for x in keyed if x[0] == p['case_id']} and
                  'G' in {x[1] for x in keyed if x[0] == p['case_id']})
                 for p in plan if p['arm'] == 'F')
    state = ('aborted_error_limit' if aborted else 'time_cap' if deadline_hit else
             'complete' if len(rows) == len(plan) else 'incomplete')
    gate = ('engineering_pilot_incomplete_or_failed' if state!='complete' or any(r['status']=='error' for r in rows) else
            'evidence_only_no_medical_answer_scoring' if mode=='evidence' else
            'blocked_no_accepted_claims' if arm_stats['G']['total_accepted_claims']==0 else
            'ready_for_double_human_adjudication_not_clinically_validated')
    return {
        'state': state, 'planned': len(plan), 'completed': len(rows),
        'paired_cases_completed': paired, 'transport_or_runtime_errors': sum(r['status']=='error' for r in rows),
        'arms': arm_stats, 'medical_accuracy': 'NOT_EVALUATED',
        'claim_verification': 'automatic quote/number checks, not semantic medical gold',
        'gate': gate,
    }


def atomic_write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description='DEV-only paired conventional RAG vs v13 grounded pilot')
    p.add_argument('--db', required=True)
    p.add_argument('--cases', default=str(ROOT / 'benchmarks' / 'pilot16_dev_questions.json'))
    p.add_argument('--aliases')
    p.add_argument('--out', required=True)
    p.add_argument('--chat-url', default='http://127.0.0.1:7866')
    p.add_argument('--qwen-url', default='http://127.0.0.1:8002/v1')
    p.add_argument('--model', default='qwen-local')
    p.add_argument('--mode', choices=('evidence', 'qwen'), default='qwen')
    p.add_argument('--judge', action='store_true')
    p.add_argument('--draft-strategy', choices=('text-first', 'slots-first', 'batch-recover'), default='text-first')
    p.add_argument('--max-draft-calls', type=int, default=6)
    p.add_argument('--max-http', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-hours', type=float, default=1.5,
                   help='Wall clock max per invocation, checked BETWEEN jobs (not a hard interrupt)')
    p.add_argument('--max-consecutive-errors', type=int, default=3)
    p.add_argument('--execute', action='store_true', help='Calls existing live services; otherwise dry-run only')
    p.add_argument('--resume', action='store_true', help='Resume same out path with identical fingerprints')
    args = p.parse_args()
    if args.max_hours <= 0 or args.max_hours > 2:
        p.error('--max-hours must be in (0,2] for the pilot; not the 10-hour experiment')
    if not 1 <= args.max_consecutive_errors <= 5:
        p.error('--max-consecutive-errors must be 1..5')
    if not 1 <= args.max_draft_calls <= 12 or not 1 <= args.max_http <= 6:
        p.error('call budgets out of range')
    if args.mode == 'evidence' and args.judge:
        p.error('judge requires qwen')
    if args.resume and not args.execute:
        p.error('--resume requires --execute')
    db = Path(args.db).resolve()
    if not db.is_file():
        p.error('SQLite DB missing')
    if args.aliases and not Path(args.aliases).is_file():
        p.error('Alias file missing')
    cases = load_cases(Path(args.cases))
    plan = make_plan(cases, args.seed)
    info = kb_info(db)
    if not args.execute:
        print(json.dumps({'dry_run': True, 'jobs': len(plan), 'cases': len(cases),
                          'kb': info, 'planned_arms': ['F', 'G'],
                          'first_tasks': plan[:6], 'uses_dev_gold': False,
                          'warning': 'Not a 2417-job experiment, E1/E2/E3 not independent arms'}, ensure_ascii=False, indent=2))
        return

    if not (ROOT / 'mhrag_v13' / 'evidence_preview.py').is_file():
        raise SystemExit('v0.5 Evidence Window is not installed; install v13_evidence_window_v05 first')
    from mhrag_v13.resolve import SQLiteStudyResolver
    from mhrag_v13.retrieve import LocalChatRetriever
    from mhrag_v13.grounded_answers import QwenJsonClient, ClaimDraftGenerator, ClaimSupportVerifier
    from mhrag_v13.grounded_controller import GroundedController

    retriever = LocalChatRetriever(args.chat_url)
    client = QwenJsonClient(args.qwen_url, args.model) if args.mode == 'qwen' else None
    if client:
        local_model_ok(args.qwen_url, args.model)
    controller = GroundedController(
        resolver=SQLiteStudyResolver(db, args.aliases), retriever=retriever, db_path=db,
        generator=ClaimDraftGenerator(client, max_calls=args.max_draft_calls, strategy=args.draft_strategy) if client else None,
        claim_verifier=ClaimSupportVerifier(client if args.judge else None), max_http_calls=args.max_http,
    )
    frozen = make_manifest(args, cases, plan, info)
    out = Path(args.out).resolve()
    lock_path = ROOT.parent / 'results' / '.v124_eval.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Another evaluation holds the shared lock: {lock_path}')
        if args.resume:
            if not out.is_dir() or not (out / 'manifest.json').is_file():
                raise SystemExit('--resume requires an existing output folder with manifest')
            old = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            if {k: old.get(k) for k in frozen} != frozen:
                raise SystemExit('RESUME REFUSED: input/config/code/database fingerprints differ')
        else:
            if out.exists():
                raise SystemExit(f'Refusing overwrite: {out}; use unique --out or --resume')
            out.mkdir(parents=True)
            atomic_write_json(out / 'manifest.json', {**frozen, 'created_utc': datetime.now(timezone.utc).isoformat()})
        rows = read_existing(out / 'results.jsonl', plan)
        next_idx = len(rows)
        deadline = time.monotonic() + args.max_hours * 3600
        errors_in_row = 0
        stopped_for_errors = False
        stopped_for_time = False
        lookup = {c['case_id']: c['question'] for c in cases}
        for prev in reversed(rows):
            if prev['status'] != 'error':
                break
            errors_in_row += 1
        if errors_in_row >= args.max_consecutive_errors:
            raise SystemExit('Last run stopped due to consecutive errors; diagnose before resuming')
        with (out / 'results.jsonl').open('a', encoding='utf-8') as f:
            for i in range(next_idx, len(plan)):
                if time.monotonic() >= deadline:
                    stopped_for_time = True
                    break
                item = run_job(plan[i], lookup, retriever, controller, args.mode)
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush(); os.fsync(f.fileno())
                rows.append(item)
                errors_in_row = errors_in_row + 1 if item['status'] == 'error' else 0
                print(f'[{len(rows)}/{len(plan)}] {item["case_id"]} {item["arm"]} '
                      f'{item["status"]} {item["wall_ms"]:.0f}ms accepted={item.get("accepted_claim_count")}', flush=True)
                if errors_in_row >= args.max_consecutive_errors:
                    stopped_for_errors = True
                    break
        report = summary(rows, plan, deadline_hit=stopped_for_time, aborted=stopped_for_errors, mode=args.mode)
        atomic_write_json(out / 'summary.json', report)
        print('END ' + json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
