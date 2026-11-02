"""Read-only, heading-aware local section audit for v13 (v3).

This module never promotes local SQLite evidence into API Gold recall, and
never attests to the clinical accuracy of a model-generated claim.

v3: handle section headings on later lines within a chunk and reject sentence
fragments (e.g. lowercase 'findings may ...') as heading evidence.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from .contracts import Source
from .verify import reference_risk

# Intentionally case-sensitive. Real article headings generally capitalize
# Results/Findings. Lowercase 'findings may not be generalized' is NOT a title.
# A leading chapter number and/or whitespace is accepted. A period after
# 'findings' is intentionally not treated as a heading delimiter.
_SECTION_HEADINGS = {
    'methods': re.compile(
        r'(?m)^[ \t]*(?:\d{1,2}[.)][ \t]*)?(?:Methods?|METHODS?|Methodology|'
        r'Study [Dd]esign(?: and [Pp]articipants)?|Participants and [Mm]ethods)'
        r'(?=[: \t\r\n]|$)'),
    'results': re.compile(
        r'(?m)^[ \t]*(?:\d{1,2}[.)][ \t]*)?(?:Results?|RESULTS?|Findings|FINDINGS)'
        r'(?=[: \t\r\n]|$)'),
}

_EXCLUDED_HEADINGS = re.compile(
    r'(?im)^[ \t]*(?:references|bibliography|acknowledg(?:e)?ments|'
    r'author contributions?|contributors?|data availability(?: statement)?|'
    r'funding|declarations? of interests?|competing interests|appendi(?:x|ces))\b')

_LATER_SECTIONS = re.compile(
    r'(?im)^[ \t]*(?:discussion|limitations?|conclusions?|general discussion|'
    r'references|bibliography|author contributions?|declarations? of interest)'
    r'(?=[: \t\r\n]|$)')
_ABSTRACT_HEADINGS = re.compile(r'(?im)^[ \t]*abstract\b(?=[: \t\r\n]|$)')


class SQLiteSectionReader:
    def __init__(self, db_path):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(self.db_path)

    def _rows(self, doc_id: int):
        if type(doc_id) is not int or doc_id < 1:
            raise ValueError('doc_id must be positive int')
        conn = sqlite3.connect('file:' + self.db_path.as_posix() + '?mode=ro', uri=True)
        try:
            return conn.execute(
                'SELECT id, ordinal, page, body FROM chunks WHERE document_id=? '
                'ORDER BY ordinal, id', (doc_id,)
            ).fetchall()
        finally:
            conn.close()

    @staticmethod
    def _excluded(text: str) -> bool:
        # Exclude chunks whose *leading section* is acknowledgements/references.
        # A separate conservative context check is also applied per candidate.
        match = _EXCLUDED_HEADINGS.search(text[:350])
        return bool(match and match.start() < 90)

    @staticmethod
    def _candidates(text: str, section: str):
        """Yield credible actual line-start headings inside a PDF text chunk."""
        for hit in _SECTION_HEADINGS[section].finditer(text):
            start = hit.start()
            # The heading can appear after a chunk's opening text; don't limit
            # it to the first 400/650 characters as v2 did.
            preceding = text[:start]
            blockers = list(_LATER_SECTIONS.finditer(preceding))
            if blockers:
                # Once Discussion/References begins, 'results' appearing later
                # in the same chunk must not be treated as primary Results.
                continue
            yield hit

    def audit_sections(self, doc_id: int, sections=('methods', 'results'), per_section: int = 1):
        """Return a heading-based, local-only original-text audit.

        `found` means a plausible section heading exists, not that medical
        facts or the full Results section have been verified.
        """
        if type(per_section) is not int or not 1 <= per_section <= 3:
            raise ValueError('per_section must be an integer between 1 and 3')
        names = tuple(str(s).lower() for s in sections)
        if not names or any(s not in _SECTION_HEADINGS for s in names):
            raise ValueError('sections must be methods and/or results')
        rows = self._rows(doc_id)
        entries = []
        for section in names:
            candidates = []
            for chunk_id, ordinal, page, body in rows:
                text = str(body or '')
                if len(text.strip()) < 70 or self._excluded(text):
                    continue
                for hit in self._candidates(text, section):
                    # If a new top-level section starts later in the same
                    # chunk, exclude those lines from the chosen evidence span.
                    end = len(text)
                    next_section = _LATER_SECTIONS.search(text, hit.end())
                    if next_section:
                        end = next_section.start()
                    excerpt = text[hit.start():end].strip()[:1800]
                    if len(excerpt) < 70:
                        continue
                    if reference_risk(Source(doc_id, int(chunk_id), str(page), '', excerpt, 'local_open')):
                        continue
                    in_abstract = (int(ordinal) < 10 or
                                   bool(_ABSTRACT_HEADINGS.search(text[:hit.start()])))
                    # Headings in the main body are preferred. A shorter
                    # abstract paragraph remains a fallback, not a substitute
                    # for body Results. This heuristic is not section proof.
                    score = 25.0 if not in_abstract else 7.0
                    score += 4 if hit.group(0).strip().lower().startswith('results') else 1
                    score += min(len(excerpt), 1200) / 400.0
                    score -= min(hit.start(), 1600) / 500.0
                    candidates.append((score, int(ordinal), int(chunk_id), str(page),
                                       excerpt, hit.start(), hit.group(0).strip(),
                                       'abstract_candidate' if in_abstract else 'body_candidate'))
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            if not candidates:
                entries.append({
                    'document_id': doc_id, 'section': section,
                    'found': False, 'status': 'heading_not_found',
                    'local_open_only': True,
                })
                continue
            selected = 0
            seen_chunk_ids = set()
            for _, ordinal, cid, page, excerpt, offset, heading, kind in candidates:
                if cid in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(cid)
                entries.append({
                    'document_id': doc_id, 'section': section,
                    'found': True, 'status': 'heading_found_not_claim_verified',
                    'chunk_id': cid, 'ordinal': ordinal,
                    'page': page, 'excerpt': excerpt[:1400],
                    'heading_offset': offset, 'heading_text': heading,
                    'section_candidate_type': kind,
                    'local_open_only': True,
                })
                selected += 1
                if selected >= per_section:
                    break
        return entries

    def open(self, doc_id: int, query: str, limit: int = 2) -> list[Source]:
        """Backward-compatible free-text reader; NOT a medical fact verifier."""
        if type(limit) is not int or not 0 <= limit <= 20:
            raise ValueError('limit must be between 0 and 20')
        tokens = {w.lower() for w in re.findall(r'[A-Za-z]{4,}', query)}
        candidates = []
        for cid, ordinal, page, body in self._rows(doc_id):
            text = str(body or '')
            if len(text.strip()) < 30 or self._excluded(text):
                continue
            src = Source(doc_id, int(cid), str(page), '', text[:1400], origin='local_open')
            score = sum(w in text[:2400].lower() for w in tokens)
            if any(list(self._candidates(text, section)) for section in _SECTION_HEADINGS):
                score += 3
            if reference_risk(src):
                score -= 15
            candidates.append((score, int(ordinal), src))
        candidates.sort(key=lambda r: (-r[0], r[1]))
        return [src for _, _, src in candidates[:limit]]
