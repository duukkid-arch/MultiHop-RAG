"""Study-aware, read-only evidence harvesting and auditable slots.

Does not use evaluation Gold IDs. Every chunk comes from the current SQLite corpus.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
import re
import sqlite3
from typing import Sequence

# Strict top-level heading patterns. PDF boundaries can occur mid-chunk.
RESULTS_HEADER = re.compile(r'(?m)^[ \t]*(?:Results|Findings|RESULTS|FINDINGS)(?=\s*[:\n]|\s+[A-Z][a-z]+\b)')
METHODS_HEADER = re.compile(r'(?im)^[ \t]*(?:methods|materials and methods|patients and methods)(?=\s*[:\n]|\s+[A-Z][a-z]+\b)')
LATER_HEADER = re.compile(r'(?im)^[ \t]*(?:discussion|limitations?|conclusions?|references|bibliography|author contributions|acknowledg(?:e)?ments)(?=\s*[:\n]|\s+[A-Z][a-z]+\b)')
PRIMARY = re.compile(r'primary\s+(?:endpoint|outcome|end\s*point)|主要终点|主要结局', re.I)
RESULT_NUMBERS = re.compile(r'(?:p\s*[=<≤≥]|95\s*%\s*(?:CI|confidence)|\bCI\b|hazard ratio|risk ratio|geometric mean|cohen|effect size|\bF\s*\(|\bRR\s*\b|\d+(?:\.\d+)?\s*(?:hours?|h|%))', re.I)
EXCLUDED = re.compile(r'(?im)^[ \t]*(?:references|bibliography|author contributions|funding|conflicts? of interest)(?=\s|:|$)')
# Discussion prose may be split from its heading across PDF chunk boundaries.
OVERVIEW_HINT = re.compile(r'(?im)\b(?:abstract|background|objective|introduction|summary|review|guideline|conclusion|study\s+aim|we\s+aim|we\s+review|purpose)\b')
DESIGN_HINT = re.compile(r'(?im)\b(?:randomi[sz]ed|double.blind|single.center|multi.center|controlled\s+trial|review|meta.analysis|guideline|cohort|observational|case.report|retrospective|systematic review)\b')
DISCUSSION_PROSE = re.compile(r'(?i)\b(?:findings? may not be generali[sz]ed|\blimitations? of (?:this|the) study|future (?:work|research) (?:is|should)|these (?:results|findings) (?:may|cannot) not)\b')

@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    document_id: int
    chunk_id: int
    ordinal: int
    page: str
    section: str
    text: str
    origin: str = 'sqlite_local_open'

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class Slot:
    document_id: int
    slot: str
    status: str
    evidence_ids: tuple[str, ...]
    reason: str

    def to_dict(self):
        d=asdict(self)
        d['evidence_ids']=list(self.evidence_ids)
        return d


def requested_slots(question: str) -> tuple[str, ...]:
    """Intent-based requirements, not automatic clinical interpretation.

    'Compare' alone does NOT imply treatment-effect data. For non-RCT
    documents, 'overview' is a grounded original-text slot rather than a
    fictitious randomized-trial Methods/Results section.
    """
    q = question.casefold()
    slots = []
    if not any(k in q for k in ('有没有报告', '是否报告')) and any(k in q for k in (
        '设计', '研究类型', '研究对象', '受试者', '人群', '年龄阶段', '随机',
        '对照', '干预', '方案', '研究方法', 'design', 'methods', 'randomized',
        'control group', 'intervention', 'population')):
        slots.append('design')
    if any(k in q for k in ('主要终点', '主要结局', 'primary endpoint', 'primary outcome')):
        slots.append('primary_outcome')
    if any(k in q for k in (
        '疗效结果', '客观注意力结果', '最终结果', '实际结果', '统计结果',
        '效应量', '效应值', '死亡率降低', 'p值', 'p 值', '置信区间',
        'result', 'efficacy', 'effect size', 'hazard ratio', 'relative risk')):
        slots.append('results')
    elif ('结果' in q or '疗效' in q or '效果' in q or '治疗效果' in q) and not any(
        k in q for k in ('有没有报告', '是否报告', '可否直接比较', '是否可以计算', '是否证明')):
        slots.append('results')
    # Non-trial evidence questions need abstract, objectives and conclusions.
    if any(k in q for k in (
        '研究问题', '研究主题', '主要发现', '主要结论', '文章', '文献',
        '综述', '指南', '共同性质', '交集', '区别', '异同', '疾病范围',
        '机制', '是否报告', '有没有报告', '是否证明', '可否直接比较',
        '是否可以计算', '能否计算', '找到', '定位', '摘要', '证据类型',
        '什么屏幕活动', '哪些屏幕活动', '什么活动类型')):
        slots.append('overview')
    if not slots:
        slots = ['overview']
    return tuple(dict.fromkeys(slots))


class SQLiteEvidenceHarvester:
    def __init__(self, db_path, *, max_evidence_per_study=8, max_text_chars=1750):
        self.db=Path(db_path).resolve()
        if not self.db.is_file():
            raise FileNotFoundError(self.db)
        if not 3 <= max_evidence_per_study <= 16:
            raise ValueError('max_evidence_per_study must be 3..16')
        self.max_evidence=max_evidence_per_study
        self.max_chars=max_text_chars

    def _rows(self, doc_id):
        if type(doc_id) is not int or doc_id < 1:
            raise ValueError('invalid document_id')
        with sqlite3.connect('file:'+self.db.as_posix()+'?mode=ro', uri=True) as db:
            return [(int(cid),int(o),str(page or ''),str(body or '')) for cid,o,page,body in db.execute(
                'SELECT id, ordinal, page, body FROM chunks WHERE document_id=? ORDER BY ordinal,id',(doc_id,))]

    def harvest(self, document_ids: Sequence[int], question: str):
        """Return spans + evidence-slot candidates. 'candidate' never means medically verified."""
        requested=requested_slots(question)
        all_spans=[];all_slots=[]
        for did in dict.fromkeys(document_ids):
            rows=self._rows(did)
            spans, slots=self._harvest_doc(did,rows,question,requested)
            all_spans.extend(spans);all_slots.extend(slots)
        return all_spans,all_slots

    def _harvest_doc(self,did,rows,question,requested):
        if not rows:
            return [],[Slot(did,s,'missing',(), 'no_document_chunks') for s in requested]
        # First choose a heading in the BODY rather than the abstract (early chunks).
        body_results=[o for _,o,_,text in rows if o>=10 and RESULTS_HEADER.search(text)]
        abstract_results=[o for _,o,_,text in rows if o<10 and RESULTS_HEADER.search(text)]
        result_start=min(body_results) if body_results else None
        # Stop at the first later major heading, treating line-anchored chapter titles only.
        result_end=None
        if result_start is not None:
            for _,o,_,text in rows:
                if o>result_start and LATER_HEADER.search(text):
                    result_end=o;break
        methods_heads=[o for _,o,_,text in rows if METHODS_HEADER.search(text)]
        methods_start=min((o for o in methods_heads if o>=10), default=None)
        selected=[]
        # Each candidate is scored for its intended evidence slot.
        for cid,ordn,page,body in rows:
            if len(body.strip())<45 or EXCLUDED.match(body.lstrip()):
                continue
            if LATER_HEADER.match(body.lstrip()) and ordn != result_start:
                continue
            # Never rank a limitations/discussion snippet as primary efficacy results.
            if DISCUSSION_PROSE.search(body[:350]) and not RESULT_NUMBERS.search(body):
                continue
            # For a chunk crossing Results->Discussion, do not use later Discussion sentences.
            if result_start is not None and ordn==result_start:
                hit=RESULTS_HEADER.search(body)
                if hit: body=body[hit.start():]
            if methods_start is not None and ordn==methods_start:
                mh=METHODS_HEADER.search(body)
                if mh and ordn != result_start: body=body[mh.start():]
            next_section=LATER_HEADER.search(body)
            if next_section:
                if next_section.start()<70:continue
                body=body[:next_section.start()]
            if len(body.strip())<45:continue
            in_results=result_start is not None and result_start<=ordn<(result_end or 999999)
            in_abstract=ordn in abstract_results
            in_methods=methods_start is not None and methods_start<=ordn<result_start if result_start is not None else False
            looks_primary=bool(PRIMARY.search(body))
            stats=len(RESULT_NUMBERS.findall(body))
            # Only use Results in the actual results span (or explicit abstract Results).
            if in_results:
                section='results'
            elif in_abstract:
                section='abstract_results'
            elif in_methods:
                section='methods'
            elif looks_primary:
                section='outcome_definition'
            elif ordn <= 18 and 'overview' in requested:
                # Review/guideline/theory papers often have no RCT Methods/Results.
                # Introductory source text remains a candidate, NOT verified proof.
                section='overview'
            elif ordn <= 18 and 'design' in requested and DESIGN_HINT.search(body):
                section='overview'
            else:
                continue
            # Penalize study flow / demographics-only results, reward effect statistics.
            score=stats*4+(8 if looks_primary else 0)+(5 if in_results else 0)
            if section=='abstract_results':score-=6
            if section=='overview':
                score=9 if OVERVIEW_HINT.search(body[:1000]) else 2
                if DESIGN_HINT.search(body):score+=3
                if ordn <= 3:score+=2
                if ordn >= 12:score-=2
                if re.search(r'(?i)\b(?:references|author contributions|acknowledg[e]?ments)\b',body[:70]):score-=20
            # A small query lexical tie breaker, excluding stop words.
            qwords={w for w in re.findall(r'[A-Za-z]{4,}',question.lower()) if w not in {'from','with','what','compare','study','results'}}
            score+=min(3,sum(w in body.lower() for w in qwords))
            selected.append((score,ordn,cid,page,section,body))
        # Reserve at least one Methods and main Results (where available), then high-value values.
        picked=[];seen=set()
        def add(group,lim):
            for item in sorted(group,key=lambda a:(-a[0],a[1],a[2])):
                if len([s for s in picked if s[4] == item[4]])>=lim:break
                if item[2] not in seen:
                    picked.append(item);seen.add(item[2])
        # Prioritize sources for requested intent. Avoid wasting all eight
        # slots on results when the question asks about study types/topics.
        if 'overview' in requested:
            add([r for r in selected if r[4]=='overview'],3)
        if 'design' in requested:
            add([r for r in selected if r[4]=='methods'],2)
        if 'primary_outcome' in requested:
            add([r for r in selected if r[4]=='outcome_definition'],2)
        if 'results' in requested:
            add([r for r in selected if r[4]=='results'],max(1,self.max_evidence-len(picked)-1))
            add([r for r in selected if r[4]=='abstract_results'],2)
        # When no actual Methods heading exists, explicitly typed study
        # overview may still ground a design-type statement in a review.
        if 'design' in requested and not any(r[4]=='methods' for r in picked):
            add([r for r in selected if r[4]=='overview'],2)
        picked=picked[:self.max_evidence]
        spans=[EvidenceSpan(f'E{did}_{cid}',did,cid,ordn,page,section,text[:self.max_chars].strip()) for _,ordn,cid,page,section,text in picked]
        slots=[]
        for slot in requested:
            if slot=='design':
                hits=[s.evidence_id for s in spans if s.section=='methods' or
                      (s.section=='overview' and DESIGN_HINT.search(s.text))]
            elif slot=='primary_outcome':
                hits=[s.evidence_id for s in spans if PRIMARY.search(s.text)]
            elif slot=='overview':
                hits=[s.evidence_id for s in spans if s.section in ('overview','abstract_results')]
            else:
                hits=[s.evidence_id for s in spans if s.section in ('results','abstract_results') and RESULT_NUMBERS.search(s.text)]
            status='candidate' if hits else 'missing'
            slots.append(Slot(did,slot,status,tuple(hits[:4]),'needs_claim_level_validation' if hits else 'no_matching_original_text'))
        return spans,slots
