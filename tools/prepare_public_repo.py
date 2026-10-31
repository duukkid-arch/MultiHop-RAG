#!/usr/bin/env python3
"""Stage a lean public GitHub snapshot while PRESERVING models, DB, raw results.

No edits/deletes of original project. Identified legacy code is excluded from
staging only. External model trees are hardlinked/copied under models/.
Oversized files are split into LFS-sized shards, with restore_large_files.py.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,sqlite3,sys
from collections import Counter
from pathlib import Path

LEGACY={'mhrag_v6_musa','mhrag_v7_8h','mhrag_evidence_guard_v02',
        'mhrag_evidence_focus_v03','mhrag_answer_repair_v04',
        'mhrag_answer_fallback_v05'}
PRUNE_DIR={'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache',
           '.venv','venv','env','node_modules','.idea','.vscode','.tox'}
PRUNE_FILE={'Thumbs.db','.DS_Store','.env','id_rsa','id_ed25519','credentials.json',
            'service-account.json','.netrc','.git-credentials'}
PRUNE_SUFFIX={'.pyc','.pyo','.swp','.swo','.tmp','.lock'}
MODEL_SOURCE={
    'qwen2.5-7b-instruct':'/root/autodl-tmp/models/Qwen2.5-7B-Instruct',
    'bge-m3':'/root/autodl-tmp/models/bge-m3',
    'bge-reranker-v2-m3':'/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'}
GITHUB_LFS_MAX=2_000_000_000 # Conservative ceiling for GitHub Free/Pro LFS (2 GB)
PATTERNS=[
 ('GitHub token',re.compile(rb'\bgh[po su]_[A-Za-z0-9]{25,}',re.I)),
 ('HuggingFace token',re.compile(rb'\bhf_[A-Za-z0-9]{24,}\b')),
 ('OpenAI-like token',re.compile(rb'\bsk-[A-Za-z0-9_-]{24,}')),
 ('AWS access key',re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
 ('private key header',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
 ('literal API secret',re.compile(rb'''(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*['\"](?!local-placeholder|example|dummy|test|\$|<|\{)[A-Za-z0-9+/_=\-]{16,}['\"]''',re.I))]

def within(path,root):
    try:path.relative_to(root);return True
    except ValueError:return False

def ignore(rel,is_dir=False):
    parts=rel.parts
    if not parts:return False
    if any(p in PRUNE_DIR for p in parts):return True
    if parts[0] in LEGACY:return True
    if parts[-1] in PRUNE_FILE:return True
    if not is_dir and (Path(parts[-1]).suffix.lower() in PRUNE_SUFFIX or parts[-1].endswith(('-wal','-shm'))):return True
    return False

def each_file(root):
    root=root.resolve()
    for parent,dirs,files in os.walk(root,topdown=True,followlinks=False):
        relparent=Path(parent).relative_to(root)
        dirs[:]=[d for d in dirs if not ignore(relparent/d,True)]
        for name in files:
            rel=relparent/name
            if not ignore(rel):yield Path(parent)/name,rel

def link_or_copy(src,dest,hardlink=False):
    dest.parent.mkdir(parents=True,exist_ok=True)
    source=src.resolve(strict=True)
    if not source.is_file():return False
    # A live knowledge base must be snapshotted through SQLite online backup;
    # a copied raw db can be inconsistent if its WAL is changing.
    if source.name=='knowledge.sqlite':
        with sqlite3.connect(f'file:{source.as_posix()}?mode=ro',uri=True) as srcdb:
            with sqlite3.connect(dest) as destdb:srcdb.backup(destdb)
        return True
    if hardlink:
        try:os.link(source,dest);return True
        except OSError:pass
    shutil.copy2(source,dest)
    return True

def tree_source_files(root):
    for parent,dirs,files in os.walk(root,topdown=True,followlinks=False):
        dirs[:]=[d for d in dirs if d not in PRUNE_DIR and d not in ('.git','__pycache__')]
        for name in files:
            if name in PRUNE_FILE or Path(name).suffix in PRUNE_SUFFIX:continue
            file=Path(parent)/name
            # Symlinks to ordinary files (HF snapshots) are resolved by link_or_copy.
            if file.is_file():yield file,file.relative_to(root)

def sha_bytes(path):
    hasher=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):hasher.update(data)
    return hasher.hexdigest()

def split_large(root,max_part):
    manifest=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.stat().st_size<=max_part:continue
        rel=path.relative_to(root)
        parts=[];dig=hashlib.sha256();i=0
        # Split into fixed-size LFS objects, record full-file sha for restoration.
        with path.open('rb') as inp:
            while True:
                chunkname=f'{path.name}.part{i:04d}'
                shard=path.with_name(chunkname)
                size=0
                with shard.open('wb') as out:
                    while size<max_part:
                        buf=inp.read(min(8*1024*1024,max_part-size))
                        if not buf:break
                        out.write(buf);dig.update(buf);size+=len(buf)
                if size==0:
                    shard.unlink();break
                parts.append(str(shard.relative_to(root)));i+=1
        if sum((root/p).stat().st_size for p in parts)!=path.stat().st_size:
            raise RuntimeError(f'分片大小不等：{path}')
        manifest.append({'original':str(rel),'parts':parts,'size':path.stat().st_size,'sha256':dig.hexdigest()})
        path.unlink() # staged hardlink only, never source
    (root/'large_files_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest

def secret_scan(root):
    findings=[]
    binary_ext={'.safetensors','.pt','.pth','.bin','.sqlite','.sqlite3','.db','.pdf','.png','.jpg','.jpeg','.zip','.part0000'}
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() in binary_ext:continue
        size=p.stat().st_size
        if size>16*1024*1024:continue # report heuristic limits clearly
        with p.open('rb') as f:b=f.read()
        if b'\x00' in b[:10000]:continue
        for name,pat in PATTERNS:
            if pat.search(b):findings.append({'file':str(p.relative_to(root)),'type':name})
    return findings

def preflight(source,out,models):
    if not source.is_dir():raise FileNotFoundError(f'项目目录不存在: {source}')
    if within(out,source):raise ValueError('发布副本必须放在项目目录之外，避免递归复制')
    files=list(each_file(source))
    missing=[name for name,path in models.items() if not Path(path).is_dir()]
    nbytes=sum(p.stat().st_size for p,_ in files)
    for name,path in models.items():
        if Path(path).is_dir():
            nbytes+=sum(x.stat().st_size for x,_ in tree_source_files(Path(path)))
    return files,missing,nbytes

def prepare(source,out,models,max_part,dry_run=False):
    source=source.resolve();out=out.resolve()
    files,missing,size=preflight(source,out,models)
    details={'source':str(source),'stage':str(out),'files_from_project':len(files),
        'bytes_estimate':size,'model_paths':models,'missing_model_paths':missing,
        'excluded_old_code':sorted(LEGACY)}
    if dry_run:return details
    if missing:raise RuntimeError('模型目录缺失，为避免不完整公开项目而中止：'+', '.join(missing))
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'发布目录不为空：{out}')
    out.mkdir(parents=True,exist_ok=True)
    for p,rel in files:link_or_copy(p,out/rel)
    for name,path in models.items():
        root=Path(path)
        for p,rel in tree_source_files(root):link_or_copy(p,out/'models'/name/rel,hardlink=True)
    # Stage the release/benchmark tooling itself so the GitHub snapshot is usable.
    kit=Path(__file__).resolve().parent.parent
    for folder in ('tools','benchmarks'):
        for helper,rel in tree_source_files(kit/folder):
            dest=out/folder/rel
            if not dest.exists():link_or_copy(helper,dest)
    # do not modify original source; do not ignore knowledge.sqlite / results / models
    prior_ignore=out/'.gitignore'
    if prior_ignore.exists():
        (out/'original_gitignore.txt').write_text(prior_ignore.read_text(encoding='utf-8',errors='replace'),encoding='utf-8')
    prior_ignore.write_text('# Only local build caches & credentials; NOT model weights, databases or raw experiments\n'
        '__pycache__/\n*.py[cod]\n.pytest_cache/\n.mypy_cache/\n.venv/\n.env\n.env.*\n'
        '.git-credentials\nid_rsa\nid_ed25519\n',encoding='utf-8')
    shards=split_large(out,max_part)
    attr=['# All model and heavy research artifacts go through Git LFS',
          'models/** filter=lfs diff=lfs merge=lfs -text',
          '*.safetensors filter=lfs diff=lfs merge=lfs -text',
          '*.pt filter=lfs diff=lfs merge=lfs -text',
          '*.pth filter=lfs diff=lfs merge=lfs -text',
          '*.bin filter=lfs diff=lfs merge=lfs -text',
          '*.sqlite filter=lfs diff=lfs merge=lfs -text',
          '*.sqlite3 filter=lfs diff=lfs merge=lfs -text',
          '*.db filter=lfs diff=lfs merge=lfs -text',
          '*.pdf filter=lfs diff=lfs merge=lfs -text',
          '*.jsonl filter=lfs diff=lfs merge=lfs -text',
          '*.zip filter=lfs diff=lfs merge=lfs -text']
    for path in out.rglob('*'):
        if path.is_file() and path.stat().st_size>45*1024*1024:
            # Explicit path covers unfamiliar large extensions (including .part0001).
            rel=path.relative_to(out).as_posix()
            if ' ' not in rel and '\t' not in rel:attr.append(rel+' filter=lfs diff=lfs merge=lfs -text')
            else:raise RuntimeError(f'大文件路径含空格，需要手工添加 Git LFS 属性: {rel}')
    (out/'.gitattributes').write_text('\n'.join(sorted(set(attr)))+'\n',encoding='utf-8')
    issues=secret_scan(out)
    (out/'PUBLIC_RELEASE_CHECK.json').write_text(json.dumps({'generated':True,'secret_pattern_hits':issues,
       'warning':'Heuristic scan only. Manually audit datasets and license before public release; no automatic publish.'},ensure_ascii=False,indent=2),encoding='utf-8')
    details['oversized_files_sharded']=len(shards)
    details['sharded_original_bytes']=sum(e['size'] for e in shards)
    details['secret_pattern_hits']=issues
    (out/'PUBLISH_NOTES.md').write_text('''# MultiHop Agentic RAG — public release snapshot

This directory is a *new staged copy* of v12.4, not the running AutoDL project.
The production application remains `mhrag_gpu_chat_musa/`, local Qwen service remains `mhrag_local_llm_musa/`.
Obsolete v6/v7 and superseded standalone guard-install modules are not staged.

**Explicitly retained:** `models/` (Qwen / BGE / CrossEncoder), `mhrag_gpu_chat_musa/data/knowledge.sqlite`, `results/` and raw benchmark files.

## Mandatory checks before public upload

1. Inspect `PUBLIC_RELEASE_CHECK.json`, search all text/logs for embedded secrets and private data. GitHub secret scanning is not complete for big pushes.
2. Confirm redistributable licenses of **all** model weights, PDFs, papers, extracted text and experimental corpora. You may need permission to publicly redistribute full research PDFs/text.
3. Check Git LFS quota/cost and per-file limits. Large files are split into pieces; after Git LFS checkout run `python tools/restore_large_files.py`.
4. Make the repository PUBLIC only after the rights and contents are approved. A public GitHub push cannot be treated as reversible confidentiality.
5. No clinical validation is claimed: document coverage and citation syntax are not factual correctness.

`tools/publish_github.sh` requires explicit `GITHUB_REPO=owner/repo` and release confirmations.
''',encoding='utf-8')
    # scripts themselves live in the source `repo_publish_kit` and should be provided
    details['estimated_lfs_gb']=round(sum(p.stat().st_size for p in (out/'models').rglob('*') if p.is_file())/1e9,2)
    return details

def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',action='append',default=[],help='NAME=/absolute/model/path; replaces defaults for NAME')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-part-mb',type=int,default=1800,help='Decimal MB, <=1900 for GitHub Free/Pro')
    a=p.parse_args()
    models=MODEL_SOURCE.copy()
    for pair in a.model:
        name,sep,path=pair.partition('=')
        if not sep or not re.fullmatch(r'[\w-]+',name):p.error('model must be NAME=/path')
        models[name]=path
    if not (100<=a.max_part_mb<=1900):p.error('max-part-mb must be 100..1900')
    try:info=prepare(a.source,a.out,models,a.max_part_mb*1_000_000,a.dry_run)
    except Exception as exc:print(f'ERROR: {exc}',file=sys.stderr);return 2
    print(json.dumps(info,ensure_ascii=False,indent=2))
    if info.get('secret_pattern_hits'):
        print('BLOCKED: 密钥疑似命中；先排查再发布',file=sys.stderr);return 3
    return 0
if __name__=='__main__':sys.exit(cli())
