"""Fail-closed reference checks and one bounded regeneration attempt.

Not a medical fact verifier. Designed to complement v0.2 + v0.3 guards.
"""
from __future__ import annotations

import os
import re

_REFS = re.compile(r'\[(\d{1,5})\]')
_EFFECT = re.compile(r'\b(aHR|HR|OR|RR)\s*[=:：]\s*(\d+(?:\.\d+)?)', re.I)
_ABSTAIN = re.compile(r'证据不足|资料不足|片段不足|未找到(?:相关)?(?:原文|证据)|无法(?:根据(?:现有|所给|这些)?(?:证据|片段|文献)|确定)|(?:not enough|insufficient) evidence', re.I)


def strict_problem(text: str, sources: list[dict]) -> str | None:
    """Require source marker for any asserted evidence answer, even if short.

    Numeric effects must be present in at least one supplied excerpt; this is a
    narrow provenance check, NOT an attribution or inferential correctness check.
    """
    if not isinstance(text, str) or not text.strip():
        return '回答为空'
    refs = [int(x) for x in _REFS.findall(text)]
    if not refs:
        # Allow *only* short, explicit abstentions, not factual answers.
        if len(text) <= 150 and _ABSTAIN.search(text) and not _EFFECT.search(text):
            return None
        return '回答没有可核查的引用编号（短答案也必须引用）'
    supported = {(kind.lower(), value) for src in sources
                 for kind, value in _EFFECT.findall(str(src.get('snippet', '')))}
    for kind, value in _EFFECT.findall(text):
        if (kind.lower(), value) not in supported:
            return f'{kind}={value} 在当前证据片段中无法核实'
    return None


def assess(text: str, question: str, sources: list[dict], base_check, evidence_check):
    return (base_check(text, len(sources))
            or strict_problem(text, sources)
            or evidence_check(question, text, sources))


def validate_and_repair(question: str, sources: list[dict], response: str,
                        answer_fn, base_check, evidence_check):
    """Return (response, error). Retry exactly once for citation formatting.

    Evidence failures cannot be silently replaced with invented citations.
    """
    issue = assess(response, question, sources, base_check, evidence_check)
    if not issue:
        return response, None
    retry_ok = os.getenv('CHAT_CITATION_RETRY', '1') == '1'
    citation_failure = ('引用' in issue or '编号' in issue)
    if not (retry_ok and citation_failure):
        return response, issue
    instruction = (
        '\n\n【引用格式修复】刚才的回答缺少有效的文献引用。'
        '请重新阅读所给原文，先用一至两句中文直接回答问题；'
        '每个由文献支持的结论后必须加一个真实引用编号，如[1]。'
        '涉及HR、aHR、OR、RR、置信区间的数值必须从原文逐字核对。'
        '如原文不足以回答某个子问题，请直接说证据不足，不得补造。'
        '不要只输出孤立数值，也不要把引用编号连续堆在开头。'
    )
    try:
        revised = answer_fn(question + instruction, sources, history=None)
    except Exception:
        return response, issue
    revised_issue = assess(revised, question, sources, base_check, evidence_check)
    return revised, revised_issue
