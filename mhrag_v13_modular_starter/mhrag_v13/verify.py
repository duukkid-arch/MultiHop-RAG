"""Structural evidence gate; explicitly DOES NOT grade medical factual accuracy."""
from __future__ import annotations
import re
from typing import Sequence
from .contracts import Assessment,Resolution,Source

REFERENCE=re.compile(r'(?im)^\s*(?:references|bibliography|参考文献)(?:\s|$)|^\s*\[\d+\]\s+\w+')

def reference_risk(source: Source) -> bool:
    t=source.snippet.strip()[:500]
    return bool(REFERENCE.search(t[:180]))

def unique_sources(sources:Sequence[Source])->list[Source]:
    seen=set();unique=[]
    for s in sources:
        key=(s.document_id, s.chunk_id if s.chunk_id is not None else (s.page,s.snippet[:120]))
        if key not in seen:
            seen.add(key);unique.append(s)
    return unique

class StructuralEvidenceGate:
    def assess(self,resolution:Resolution,sources:Sequence[Source],question:str)->Assessment:
        observed=tuple(sorted({s.document_id for s in sources if s.origin=='api'}))
        suspected=tuple(sorted({d for d in observed if not any(s.document_id==d and
                   len(s.snippet.strip())>=30 and not reference_risk(s) for s in sources)}))
        if resolution.unresolved:
            status='UNRESOLVED_STUDY_IDENTITY'
        elif resolution.complete:
            missing=tuple(sorted(set(resolution.ids)-set(observed)))
            if missing: status='MISSING_REQUIRED_STUDIES'
            elif set(suspected)&set(resolution.ids):status='REFERENCE_ONLY_OR_THIN_EVIDENCE'
            else:status='STUDIES_LOCATED_FACTS_UNVERIFIED'
        elif not sources:
            status='NO_EVIDENCE'
        else:
            status='UNIDENTIFIED_TARGET_FACTS_UNVERIFIED'
        missing=tuple(sorted(set(resolution.ids)-set(observed)))
        return Assessment(status,resolution.ids,observed,missing,suspected,False)
