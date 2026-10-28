"""Conservative check for a misleading 'no effect data' answer when an aHR and CI are in the retrieved evidence.

This is a partial guardrail, not a medical entailment or factuality verifier.
"""
from __future__ import annotations
import re

_EFFECT_Q = re.compile(r'复发|住院|风险|疗效|作用|差异|相关|\b(?:aHR|hazard ratio|outcome|effect)\b',re.I)
_MEASURED = re.compile(r'\b(?:aHR|adjusted hazard ratio|HR)\s*[=:：]\s*\d+(?:\.\d+)?',re.I)
_CI = re.compile(r'95\s*%\s*C[Ii]|置信区间',re.I)
_FALSE_DENIAL = re.compile(
    r'(?:未|没有|并未|不曾|无法).{0,12}(?:详细)?(?:描述|报告|提供|发现|说明|确定).{0,60}'
    r'(?:复发|住院|风险|疗效|差异|关系|关联|效果)',re.I)
_NUMERIC_REPLY = re.compile(r'\b(?:aHR|HR|OR|RR)\s*[=:：]\s*\d+(?:\.\d+)?',re.I)

def evidence_problem(question: str, answer: str, sources: list[dict]) -> str | None:
    """Reject the *specific* false-denial pattern seen in the RAG demo.

    General claims, statistic attribution, and safety outcomes still require human review.
    """
    if not _EFFECT_Q.search(question):
        return None
    evidence='\n'.join(str(s.get('snippet','')) for s in sources)
    if _MEASURED.search(evidence) and _CI.search(evidence):
        if _FALSE_DENIAL.search(answer) and not _NUMERIC_REPLY.search(answer):
            return '原文含有风险比及置信区间，但生成结果称未报告疗效或住院风险，疑似误读证据'
    return None
