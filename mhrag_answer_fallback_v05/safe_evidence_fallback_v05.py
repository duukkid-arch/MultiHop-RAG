"""Verbatim, cited excerpts when a local LLM answer fails validation.

Deliberately does NOT cite, paraphrase or rehabilitate the failed model output.
Only quotes exact retrieved source substrings with stable source IDs.
"""
from __future__ import annotations

import re

_EFFECT = re.compile(r'\b(?:aHR|adjusted hazard ratio|HR|OR|RR)\s*[=:：]\s*\d+(?:\.\d+)?', re.I)
_CI = re.compile(r'95\s*%\s*CI\s*[:：]?\s*\d+(?:\.\d+)?\s*[–—-]\s*\d+(?:\.\d+)?', re.I)


def _verbatim_window(raw: str, maximum: int = 460) -> str:
    """Return a true substring, preferring a quantitative result if present."""
    raw = str(raw or '').strip()
    if not raw:
        return '（本条证据正文为空）'
    quantitative = _EFFECT.search(raw)
    if quantitative:
        # Shift toward the beginning of the sentence but retain trailing CI.
        start = max(0, quantitative.start() - 275)
        end = min(len(raw), start + maximum)
        # Ensure the CI appears if the effect and CI are close together.
        ci = _CI.search(raw, quantitative.start())
        if ci and ci.end() <= min(len(raw), quantitative.start() + 340) and ci.end() > end:
            end = min(len(raw), ci.end() + 45)
            start = max(0, end - maximum)
        excerpt = raw[start:end]
        return ('…' if start else '') + excerpt + ('…' if end < len(raw) else '')
    return raw[:maximum] + ('…' if len(raw) > maximum else '')


def extractive_fallback(sources: list[dict], issue: str | None = None, max_sources: int = 2) -> str:
    """Make a safe evidence-first fallback; never incorporate the model's claims.

    Source citation numbers are copied from retrieval, not fabricated. It does not
    claim that a passage proves the answer. Clinical inference is left open.
    """
    lines = [
        '【原文摘录模式】本地模型未能生成通过引用核验的完整回答。',
        '下面仅展示检索到的原文片段，不代表模型已经得出可靠的医学结论；请核对 PDF 原文。',
    ]
    if not sources:
        lines.append('当前没有可用的原文片段。')
        return '\n'.join(lines)
    used = 0
    for idx, src in enumerate(sources):
        if used >= max(1, int(max_sources)):
            break
        citation = src.get('citation', idx + 1)
        if not isinstance(citation, int) or citation < 1:
            citation = idx + 1
        title = str(src.get('title', '未知文献')).replace('\n', ' ').strip()[:160]
        page = src.get('page')
        page_text = str(page) if isinstance(page, int) and page > 0 else '未知'
        snippet = _verbatim_window(src.get('snippet', ''))
        lines.append(f'[{citation}] {title}（PDF 第 {page_text} 页）原文：“{snippet}”')
        used += 1
    lines.append('这些片段可能不足以回答全部子问题（尤其是安全性、因果关系和未报告的结局），不能自行补充结论。')
    return '\n\n'.join(lines)
