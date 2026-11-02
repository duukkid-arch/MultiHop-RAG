"""Read-only study identity resolver. Works only on unlabeled corpus metadata.

CAUTION: heuristic aliases are *not* a medical ontology and cannot infer missing
study names from the hidden answer key. Ambiguous names remain unresolved.
"""
from __future__ import annotations
import json
import re
import sqlite3
from pathlib import Path
from .contracts import Study, Resolution

BLOCKED={'RAG','LLM','PDF','BMI','CI','HR','MRI','CT','RCT','USA','ACS',
         'BM25','AI','GPT','COVID','DOI','BGE','CE','PICO','MPH','WHO',
         'ADHD','TOVA','CGI','SNAP','SNAP-IV','IMV','SLE','LRSQ','RCT'}
COMPARE=re.compile(r'比较|对比|相比|异同|vs\.?\b|versus|compare|comparison|两篇|两项|分别',re.I)
ACRONYM=re.compile(r'(?<![a-zA-Z0-9])([A-Z][A-Z0-9-]{2,})(?![a-zA-Z0-9])')
MIXED=re.compile(r'(?<![A-Za-z])([A-Z][A-Za-z0-9-]+(?:\s+(?:and|of|the|in|on|for|with|[A-Z][A-Za-z0-9-]+)){1,5})(?![A-Za-z])')
PDF=re.compile(r'(?<![A-Za-z0-9_])([A-Za-z0-9][A-Za-z0-9_.-]*\.pdf)\b',re.I)


def anchors(question: str) -> tuple[str,...]:
    titles=re.findall(r'《([^》]{3,100})》',question)
    unquoted=re.sub(r'《[^》]{3,100}》',' ',question)
    filenames=PDF.findall(unquoted)
    rest=PDF.sub(' ',unquoted)
    raw=[*titles,*filenames,*ACRONYM.findall(rest),*MIXED.findall(rest)]
    found=[]
    for value in raw:
        name=' '.join(value.split()).strip(',.()')
        if len(name)<3 or name.upper() in BLOCKED or name.casefold() in [x.casefold() for x in found]:
            continue
        found.append(name)
    return tuple(found[:8])

class SQLiteStudyResolver:
    def __init__(self, db_path: str | Path, alias_path: str | Path | None = None):
        self.db_path=Path(db_path).resolve()
        if not self.db_path.is_file(): raise FileNotFoundError(self.db_path)
        conn=sqlite3.connect('file:'+self.db_path.as_posix()+'?mode=ro',uri=True)
        try:
            docs=conn.execute('SELECT id, filename FROM documents').fetchall()
            heads={}
            for did,body in conn.execute('SELECT document_id,body FROM chunks WHERE ordinal < 3 ORDER BY document_id,ordinal'):
                did=int(did)
                heads[did]=(heads.get(did,'')+' '+str(body or '')[:1000])[:3000]
        finally:
            conn.close()
        self.docs=[(int(i),str(fn),heads.get(int(i),'')) for i,fn in docs]
        self.doc_names={did:title for did,title,_ in self.docs}
        self.aliases={}
        if alias_path is not None:
            self._load_verified_aliases(Path(alias_path))

    def _load_verified_aliases(self, alias_path: Path) -> None:
        """Optional, manually checked aliases. Never infer IDs from evaluation Gold labels.

        Two protections: explicit source-review attestation and a matching literal
        phrase in the selected corpus document. The latter is a sanity check,
        NOT proof that the paper is the original study; PDF review is required.
        """
        data=json.loads(alias_path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or not isinstance(data.get('aliases'),list):
            raise ValueError('Alias manifest must be an object with an aliases list')
        conn=sqlite3.connect('file:'+self.db_path.as_posix()+'?mode=ro',uri=True)
        try:
            for row in data['aliases']:
                if not isinstance(row,dict): raise ValueError('Alias entry must be an object')
                alias=row.get('alias')
                did=row.get('document_id')
                phrase=row.get('source_phrase')
                verified=row.get('verified_against_source')
                if not isinstance(alias,str) or len(alias.strip())<3:
                    raise ValueError('Alias must be at least 3 characters')
                if type(did) is not int or did not in self.doc_names:
                    raise ValueError(f'Unknown document_id for alias {alias!r}: {did!r}')
                if not isinstance(phrase,str) or len(phrase.strip())<3:
                    raise ValueError(f'Missing source_phrase for {alias!r}')
                if verified is not True:
                    continue  # Draft aliases NEVER alter the retrieval decision
                key=' '.join(alias.split()).casefold()
                if key in self.aliases:
                    raise ValueError(f'Duplicate verified alias: {alias!r}')
                filename=self.doc_names[did]
                phrase_ok=phrase.casefold() in filename.casefold()
                if not phrase_ok:
                    phrase_ok=conn.execute(
                        'SELECT 1 FROM chunks WHERE document_id=? AND '
                        'instr(lower(body),lower(?))>0 LIMIT 1',(did,phrase)
                    ).fetchone() is not None
                if not phrase_ok:
                    raise ValueError(f'Source phrase not present in document {did} for {alias!r}')
                self.aliases[key]=(did,filename)
        finally:
            conn.close()

    def candidates(self, term:str, limit:int=5)->list[dict]:
        needle=' '.join(term.split()).casefold()
        pat=re.compile(r'(?<![a-z0-9])'+re.escape(needle)+r'(?![a-z0-9])')
        ranks=[]
        for did,title,head in self.docs:
            in_name=bool(pat.search(title.casefold()))
            in_head=bool(pat.search(' '.join(head[:2000].split()).casefold()))
            score=(12 if in_name else 0)+(5 if in_head else 0)
            if score:ranks.append({'document_id':did,'title':title,'score':score})
        ranks.sort(key=lambda row:(-row['score'],row['document_id']))
        return ranks[:limit]

    def resolve(self, question: str) -> Resolution:
        terms=anchors(question)
        studies=[];missing=[]
        for term in terms:
            key=' '.join(term.split()).casefold()
            if key in self.aliases:
                did,title=self.aliases[key]
                studies.append(Study(did,title,term,20.0))
                continue
            ranks=self.candidates(term,limit=2)
            if not ranks or (len(ranks)>1 and ranks[0]['score']-ranks[1]['score']<4):
                missing.append(term)
            else:
                best=ranks[0]
                studies.append(Study(best['document_id'],best['title'],term,float(best['score'])))
        # Study comparisons with one known and another unnamed entity must NOT be scoped.
        if COMPARE.search(question) and terms and len({s.document_id for s in studies})<2:
            missing.append('unresolved_comparison_study')
        return Resolution(tuple(studies),tuple(dict.fromkeys(missing)),terms)
