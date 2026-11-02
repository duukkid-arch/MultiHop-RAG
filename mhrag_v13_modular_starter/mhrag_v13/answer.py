"""Safe default answer: show evidence without claiming automatic clinical validation."""
from __future__ import annotations
from typing import Sequence
from .contracts import Source,Assessment

class EvidencePacketAnswerer:
    def compose(self,question:str,sources:Sequence[Source],assessment:Assessment,
                fallback_answer:str='')->str:
        if not sources:
            return '没有找到可供核验的医学论文证据。请确认研究名称或 PDF 内容。'
        lines=[f'证据状态：{assessment.status}。以下是检索证据摘要，并非自动核验后的医学结论。']
        if assessment.missing_ids:
            lines.append('尚缺论文 ID：'+', '.join(map(str,assessment.missing_ids)))
        for i,s in enumerate(sources[:8],1):
            lines.append(f'[{i}] 文献 {s.document_id}，第 {s.page or "?"} 页：{s.snippet[:450].strip()}')
        lines.append('医学数值和结论需结合原文 Methods/Results 与具体引用逐条复核。')
        return '\n\n'.join(lines)
