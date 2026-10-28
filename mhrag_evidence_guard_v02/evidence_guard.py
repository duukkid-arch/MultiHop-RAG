"""Conservative, transparent evidence gates for a local medical-paper RAG demo.

This is a rule-based safety backstop, NOT a calibrated relevance classifier.
Unknown topics pass through; do not mistake this for clinical-grade verification.
"""
from __future__ import annotations
import re

_CLOZ = re.compile(r'clozapine|氯氮平', re.I)
_DEPRESSION = re.compile(r'抑郁症|重性抑郁|major depressive disorder|\bMDD\b|depression', re.I)
_CAUSES_QUESTION = re.compile(r'病因|致病|发病机制|病理生理|成因|危险因素|etiolog|pathogen|causes? of|risk factors? for',re.I)
_CAUSES_EVIDENCE = re.compile(
    r'etiolog(?:y|ical)|pathogenesis|pathophysiolog|'
    r'cause(?:s|d)? of (?:major )?depression|'
    r'risk factors? (?:for|of) (?:major )?depression|'
    r'depression (?:is caused|results from|develops due)|'
    r'抑郁症.{0,10}(?:病因|病理生理|致病|发病机制|危险因素)|'
    r'(?:病因|致病|发病机制).{0,10}抑郁症',re.I)
_REF = re.compile(r'\[(\d{1,6})\]')

def topic_relevant(question: str, title: str, snippet: str) -> bool:
    content = (title or '') + '\n' + (snippet or '')
    # For explicitly named, high-stakes drug questions, do not use unrelated papers.
    if _CLOZ.search(question) and not _CLOZ.search(content):
        return False
    # A question about etiology cannot be answered using treatment-efficacy-only paragraphs.
    if _DEPRESSION.search(question) and _CAUSES_QUESTION.search(question):
        return bool(_DEPRESSION.search(content) and _CAUSES_EVIDENCE.search(content))
    return True

def citation_problem(text: str, n_sources: int) -> str | None:
    ids = [int(x) for x in _REF.findall(text)]
    if len(ids) > 16:
        return '引用数量异常（疑似模型重复生成）'
    if any(x < 1 or x > n_sources for x in ids):
        return '引用编号超出本次提供的证据范围'
    if any(ids.count(i)>5 for i in set(ids)):
        return '同一引用被异常重复'
    # A long purported evidence answer with no source markers is not auditable.
    if len(text) >= 80 and not ids and not re.search(r'证据不足|无法(?:根据|确定)|未找到|没有.*证据|cannot (?:determine|conclude)', text, re.I):
        return '回答没有提供可核查的文献引用'
    return None
