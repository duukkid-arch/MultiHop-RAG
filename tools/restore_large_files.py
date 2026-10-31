#!/usr/bin/env python3
"""Reassemble Git LFS chunks after clone. Never overwrites existing originals."""
import hashlib,json,sys
from pathlib import Path

def restore(root):
    root=Path(root).resolve();m=root/'large_files_manifest.json'
    if not m.exists():return 0
    entries=json.loads(m.read_text(encoding='utf-8'))
    for x in entries:
        out=(root/x['original']).resolve()
        if root not in out.parents:raise RuntimeError('unsafe output path')
        if out.exists():
            # avoid accidentally overwriting a locally edited model file
            h=hashlib.sha256(out.read_bytes()).hexdigest() if out.stat().st_size<64*1024*1024 else None
            if h is not None and h==x['sha256']:continue
            raise FileExistsError(f'file exists, refusing overwrite: {out}')
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp=out.with_name(out.name+'.reassembling');d=hashlib.sha256();total=0
        try:
            with tmp.open('wb') as f:
                for part in x['parts']:
                    src=(root/part).resolve()
                    if root not in src.parents or not src.is_file():raise FileNotFoundError(part)
                    with src.open('rb') as inp:
                        while True:
                            b=inp.read(8*1024*1024)
                            if not b:break
                            total+=len(b);d.update(b);f.write(b)
            if total!=x['size'] or d.hexdigest()!=x['sha256']:
                raise RuntimeError('sha256/size mismatch: '+x['original'])
            tmp.replace(out);print('restored',x['original'])
        except Exception:
            tmp.unlink(missing_ok=True);raise
    return len(entries)
if __name__=='__main__':
    print('restored shards:',restore(sys.argv[1] if len(sys.argv)>1 else '.'))
