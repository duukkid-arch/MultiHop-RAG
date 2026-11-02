#!/usr/bin/env python3
"""Download BigBio MedHop SOURCE-schema parquet mirror, preserving per-case supports.

Use this only for the original public MedHop train/validation splits; no private gold.
No import dependency until local parquet decoding is needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

DATASET = 'bigbio/medhop'
CONFIG = 'medhop_source'  # Do not switch to medhop_bigbio_qa: it flattens supports.
EXPECTED = {'train': 1620, 'validation': 342}
DEFAULT_ENDPOINTS = ['https://huggingface.co', 'https://hf-mirror.com']


def fetch_json(url, timeout=25):
    request = urllib.request.Request(url, headers={'User-Agent':'MultiHop-RAG-MedHop-Research/1.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parquet_urls(catalog):
    """Parse HF /api/datasets/.../parquet nested split => [url] map.

    Refuse flattened bigbio_qa, because preserving per-document supports matters.
    """
    if not isinstance(catalog, dict) or CONFIG not in catalog:
        raise ValueError(f'Missing {CONFIG} parquet files; must preserve per-document supports')
    subset = catalog[CONFIG]
    if not isinstance(subset, dict):
        raise ValueError(f'Invalid {CONFIG} split map')
    files = {}
    for split in ('train', 'validation'):
        candidates = subset.get(split)
        if not isinstance(candidates, list) or not candidates or not all(isinstance(u, str) and u.startswith('https://') for u in candidates):
            raise ValueError(f'Missing/invalid {CONFIG}/{split} files')
        files[split] = candidates
    return files


def mirror_url(url, endpoint):
    """Swap host only for public Hugging Face file URLs, not arbitrary websites."""
    p = urllib.parse.urlsplit(url)
    if p.netloc not in ('huggingface.co', 'hf-mirror.com'):
        return url
    base = urllib.parse.urlsplit(endpoint)
    return urllib.parse.urlunsplit((base.scheme, base.netloc, p.path, p.query, p.fragment))


def save_url(url, dest, endpoint):
    """Atomic download so failures cannot be mistaken for complete parquet files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size:
        return 'cached'
    tmp = dest.with_suffix(dest.suffix + '.part')
    tmp.unlink(missing_ok=True)
    request = urllib.request.Request(mirror_url(url, endpoint), headers={'User-Agent':'MultiHop-RAG-MedHop-Research/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, tmp.open('wb') as fout:
            shutil.copyfileobj(response, fout, length=1 << 20)
        if not tmp.stat().st_size:
            raise IOError('downloaded empty parquet file')
        tmp.replace(dest)
        return 'downloaded'
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def convert_parquet_to_original(raw_files, dest):
    """Require schema equality and case counts; never silently flatten evidence."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('pyarrow not found. Run setup_medhop_autodl.sh; it creates an isolated venv and installs pyarrow.') from exc

    dest.mkdir(parents=True, exist_ok=True)
    output_names = {'train': 'train.json', 'validation': 'dev.json'}
    result = {}
    for split in ('train', 'validation'):
        out = dest / output_names[split]
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite existing source JSON: {out}')
        rows = []
        for source_file in raw_files[split]:
            table = pq.read_table(source_file)
            missing = {'id', 'query', 'candidates', 'supports', 'answer'} - set(table.column_names)
            if missing:
                raise ValueError(f'{source_file}: missing source columns {missing}')
            for r in table.select(['id', 'query', 'candidates', 'supports', 'answer']).to_pylist():
                rows.append(r)
        if len(rows) != EXPECTED[split]:
            raise ValueError(f'{split}: expected {EXPECTED[split]} rows, got {len(rows)}')
        for i, r in enumerate(rows):
            if not isinstance(r['supports'], list) or len(r['supports']) == 0 or not all(isinstance(s, str) for s in r['supports']):
                raise ValueError(f'{split}/{i}: supports are not preserved as a list of documents')
            if not isinstance(r['candidates'], list) or r['answer'] not in r['candidates']:
                raise ValueError(f'{split}/{i}: invalid answer/candidates')
        result[split] = rows
    # No writes until both splits pass every check.
    for split, rows in result.items():
        path = dest / output_names[split]
        with path.open('x', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False)
    return {split: len(rows) for split, rows in result.items()}


def download_and_convert(workdir, endpoints):
    if (workdir/'medhop_original'/'train.json').exists() or (workdir/'medhop_original'/'dev.json').exists():
        raise FileExistsError('medhop_original already contains files; do not overwrite. Reuse with prepare_medhop.py.')
    files = None
    chosen = None
    errors = []
    for endpoint in endpoints:
        try:
            catalog = fetch_json(endpoint.rstrip('/') + '/api/datasets/' + DATASET + '/parquet')
            files = parquet_urls(catalog)
            chosen = endpoint
            print('PARQUET_CATALOG_OK:',endpoint)
            break
        except Exception as e:
            errors.append(f'{endpoint}: {type(e).__name__}: {str(e)[:150]}')
    if files is None:
        raise RuntimeError('Hugging Face and mirror unavailable or SOURCE parquet absent: ' + '; '.join(errors))
    source_files = {}
    for split in ('train','validation'):
        source_files[split] = []
        for index, url in enumerate(files[split]):
            dst = workdir/'parquet_cache'/split/f'{index:04d}.parquet'
            if not dst.is_file() or not dst.stat().st_size:
                failed = []
                for endpoint in [chosen] + [x for x in endpoints if x != chosen]:
                    try:
                        print('DOWNLOADING:',split,index,'via',endpoint,flush=True)
                        save_url(url,dst,endpoint)
                        break
                    except Exception as e:
                        failed.append(f'{endpoint}: {type(e).__name__}: {str(e)[:150]}')
                else:
                    raise RuntimeError(f'{split} shard {index} inaccessible: '+ '; '.join(failed))
            source_files[split].append(dst)
    counts = convert_parquet_to_original(source_files, workdir/'medhop_original')
    metadata = {
        'dataset': DATASET, 'config':CONFIG, 'selected_catalog_endpoint':chosen,
        'transport':'Hugging Face auto-converted parquet; source schema',
        'note':'Public MedHop train and validation; NOT a private sealed holdout',
        'source_files': {split:[{'file':str(p.relative_to(workdir)), 'sha256':sha256(p)} for p in ps] for split,ps in source_files.items()},
        'counts':counts,
        'license':'CC BY-SA 3.0; attribution and share-alike apply to dataset',
        'upstream':'https://qangaroo.cs.ucl.ac.uk/',
    }
    (workdir/'download_manifest.json').write_text(json.dumps(metadata, indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, default=Path(os.environ.get('MEDHOP_WORKDIR','/root/autodl-tmp/medhop_benchmark_private')))
    parser.add_argument('--endpoint', action='append',help='HF endpoint(s) to try, e.g. https://hf-mirror.com')
    args=parser.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    endpoints=args.endpoint or ([os.environ['HF_ENDPOINT']] if os.environ.get('HF_ENDPOINT') else []) + DEFAULT_ENDPOINTS
    endpoints=list(dict.fromkeys(e.rstrip('/') for e in endpoints))
    try:
        result=download_and_convert(args.workdir,endpoints)
        print('MEDHOP_SOURCE_READY:',args.workdir/'medhop_original',result['counts'])
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit('DOWNLOAD_REFUSED: '+str(exc))

if __name__=='__main__': main()
