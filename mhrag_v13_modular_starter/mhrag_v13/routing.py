"""Conservative route choice for v13. Never use evaluation labels.

An API-retrieved document is a *provisional source candidate*, not an
identified/verified paper. A source ID is trusted only after the matching
(doc_id, chunk_id) pair exists in the read-only SQLite corpus.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class Route:
    mode: str
    reason: str
    document_ids: tuple[int, ...] = ()
    provisional: bool = False
    def to_dict(self) -> dict:
        return {'mode': self.mode, 'reason': self.reason,
                'document_ids': list(self.document_ids),
                'provisional': self.provisional,
                'named_studies_complete': self.mode == 'confirmed_named'}


def decide_route(identity, api_sources: list[dict], db_path: str | Path,
                 *, max_provisional_docs: int = 2) -> Route:
    if identity.complete:
        return Route('confirmed_named', 'all_named_studies_resolved', identity.ids)
    if identity.ids:
        # Do not fabricate a missing comparison paper: only harvest named IDs.
        return Route('partial_named', 'some_named_studies_unresolved', identity.ids)
    if identity.unresolved:
        return Route('blocked_named', 'unresolved_explicit_study_name')
    if identity.anchors:
        return Route('blocked_named', 'unmatched_named_study_anchor')

    # No study name was actually supplied; this is an unscoped topic query.
    # Avoid arbitrary first-result routing. At least two distinct, locally
    # verifiable API chunk IDs per candidate document are required.
    candidates = Counter()
    distinct = {}
    for source in api_sources:
        did = source.get('document_id')
        cid = source.get('chunk_id')
        if type(did) is int and did > 0 and type(cid) is int and cid > 0:
            distinct.setdefault(did, set()).add(cid)
    if not distinct:
        return Route('blocked_no_anchor', 'no_named_studies_or_locally_checkable_api_sources')
    db = Path(db_path).resolve()
    with sqlite3.connect('file:' + db.as_posix() + '?mode=ro', uri=True) as con:
        for did, cids in distinct.items():
            if len(cids) < 2:
                continue
            marks = ','.join('?' for _ in cids)
            matches = con.execute('SELECT DISTINCT id FROM chunks '
                                  f'WHERE document_id=? AND id IN ({marks})',
                                  (did, *sorted(cids))).fetchall()
            if len(matches) >= 2:
                candidates[did] = len(matches)
    if not candidates:
        return Route('blocked_no_anchor', 'insufficient_independent_api_chunks_in_local_corpus')
    ids = tuple(did for did, n in sorted(candidates.items(),
                  key=lambda x: (-x[1], x[0]))[:max_provisional_docs])
    return Route('provisional_api', 'unscoped_topic_derived_from_verified_api_chunk_ids',
                 ids, True)
