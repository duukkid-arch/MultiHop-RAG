#!/usr/bin/env python3
"""Stop only a verified v6 run tree, releasing GPU, then resume with v6's own checkpoints.

This is a STOP/RESTART workflow, NOT SIGSTOP (which keeps VRAM allocated).
A interrupted JSONL line must be checked with verify_checkpoint.py before resume.
"""
import argparse,datetime,os,signal,subprocess,sys,time
from pathlib import Path

DEFAULT='results/v6_focus_750_musa'

def cmdline(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:return ''

def processes():
    out={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            body=(p/'stat').read_text();stat=body.rsplit(')',1)[1].split()
            out[int(p.name)]=int(stat[1])
        except (OSError,IndexError,ValueError):pass
    return out

def descendants(root):
    ppids=processes();found=[];pending=[root]
    while pending:
        parent=pending.pop()
        children=[pid for pid,ppid in ppids.items() if ppid==parent]
        found.extend(children);pending.extend(children)
    return found

def go():
    pa=argparse.ArgumentParser()
    pa.add_argument('--pid',type=int,default=79544)
    pa.add_argument('--root',default=DEFAULT)
    pa.add_argument('--stop',action='store_true')
    args=pa.parse_args()
    p=Path(args.root)
    status=(p/'STATUS').read_text().strip() if (p/'STATUS').exists() else 'UNKNOWN'
    name=cmdline(args.pid)
    print('STATUS:',status,'\nPID:',args.pid,'\nCOMMAND:',name or 'not running')
    kids=descendants(args.pid)
    for pid in kids:print('CHILD:',pid,cmdline(pid)[:220])
    if not args.stop:return
    if status=='DONE':print('Already DONE. No termination needed.');return
    if not name:
        print('PID does not exist; do not force kill anything.');return
    if 'mhrag_v6_musa/run.py' not in name:
        raise SystemExit('Refusing: PID is not the expected v6 launcher.')
    if not (p/'config.json').exists():raise SystemExit('Refusing: v6 output root not found')
    # STOP the parent to eliminate the race where a new child starts after listing descendants.
    os.kill(args.pid,signal.SIGSTOP)
    try:
        kids=descendants(args.pid)
        # Only stop subprocesses descended from this specific launcher, never other GPU jobs.
        for pid in reversed(kids):
            try:os.kill(pid,signal.SIGTERM);print('SIGTERM child',pid)
            except ProcessLookupError:pass
    finally:
        try:os.kill(args.pid,signal.SIGCONT)
        except ProcessLookupError:pass
    # Parent should notice the interrupted step, set STATUS FAILED and exit.
    for _ in range(50):
        if not cmdline(args.pid):break
        time.sleep(.2)
    if cmdline(args.pid):
        print('WARNING: launcher remains alive; inspect process tree manually. Not force-killing.',file=sys.stderr)
    else:print('Launcher exited; GPU task should have released VRAM.')
    print('FINAL STATUS:',(p/'STATUS').read_text().strip() if (p/'STATUS').exists() else 'unknown')
    print('The checkpointed scoring files were NOT deleted. Check tools/verify_checkpoint.py before resume.')

if __name__=='__main__':go()
