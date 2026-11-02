"""Adaptive + study-aware corrective orchestration, capped HTTP budget.

This is a standalone integration adapter: it does not patch the running server.
Evidence verification is structural only. Claims need human/independent checking.
"""
from __future__ import annotations
import re
import sqlite3
import time
from .contracts import Resolver,Retriever,EvidenceVerifier,Answerer,TraceSink,SectionReader
from .verify import unique_sources
from .tracing import NullTrace

COMPARE=re.compile(r'比较|对比|相比|分别|两篇|两项|vs\.?\b|versus|compare',re.I)
COMPLEX=re.compile(r'为什么|如何|机制|效应量|效果|疗效|终点|差异|综合|证据|数字|风险比|亚组|优劣',re.I)

class AdaptiveController:
    def __init__(self, resolver:Resolver, retriever:Retriever, verifier:EvidenceVerifier,
                 answerer:Answerer, *, section_reader:SectionReader|None=None,
                 trace:TraceSink|None=None, max_http_calls:int=4):
        if not 1<=max_http_calls<=6:raise ValueError('HTTP budget must be 1–6')
        self.resolver,self.retriever,self.verifier,self.answerer=resolver,retriever,verifier,answerer
        self.reader=section_reader;self.trace=trace or NullTrace();self.budget=max_http_calls

    def run(self,question:str)->dict:
        if not isinstance(question,str) or not 1<=len(question.strip())<=2500:
            raise ValueError('question length must be 1–2500')
        started=time.monotonic()
        identity=self.resolver.resolve(question)
        sources=[]; attempts=[]; section_audit=[]; fallback_answer=''; calls=0
        def call(q:str, agentic=False, ids=None, action='rag'):
            nonlocal calls,fallback_answer
            if calls>=self.budget:return False
            try:
                response=self.retriever.search(q,agentic=agentic,document_ids=ids)
                before=len(sources)
                sources[:]=unique_sources([*sources,*response.sources])
                if not fallback_answer: fallback_answer=response.answer
                gained=len(sources)-before
                attempts.append({'action':action,'agentic':agentic,'scope':list(ids) if ids is not None else None,
                                 'new_sources':gained,'latency_ms':response.elapsed_ms,
                                 'status':'ok','planner_errors':response.trace.get('planner_errors',response.trace.get('planner_failures',0))})
                self.trace.emit({'type':'tool','action':action,'new_sources':gained,'elapsed_ms':response.elapsed_ms,
                                 'returned_docs':sorted({s.document_id for s in response.sources})})
                calls+=1
                return True
            except Exception as exc:
                calls+=1
                attempts.append({'action':action,'status':'error','error_type':type(exc).__name__})
                self.trace.emit({'type':'error','action':action,'error_type':type(exc).__name__})
                return False
        # First: cheap broad scout; don't assume Agentic helps when evidence already suffices.
        call(question,False,None,'rag_scout')
        assessment=self.verifier.assess(identity,sources,question)
        complex_query=bool(COMPLEX.search(question) or COMPARE.search(question))
        # Targeted retrieval: only when ALL named papers are confidently mapped.
        if identity.complete and assessment.missing_ids:
            for did in assessment.missing_ids:
                if calls>=self.budget:break
                title=next((s.title for s in identity.studies if s.document_id==did),'目标论文')
                scoped_q=f'请从论文《{title}》中检索与问题相关的 Methods/Results 原文证据；若不存在结果请说明。问题：{question}'
                call(scoped_q,False,[did],'targeted_paper_search')
            assessment=self.verifier.assess(identity,sources,question)
        # Complex unscoped/missing cases may benefit from multi-step agent; hard budget.
        if (complex_query and (not assessment.structurally_complete) and calls<self.budget):
            call(question,True,None,'agent_gap_search')
            assessment=self.verifier.assess(identity,sources,question)
        # One more targeted attempt only if real documented gap persists.
        if identity.complete and assessment.missing_ids and calls<self.budget:
            for did in assessment.missing_ids:
                if calls>=self.budget:break
                call(question,False,[did],'last_gap_search')
            assessment=self.verifier.assess(identity,sources,question)
        if self.reader and identity.complete:
            for did in identity.ids[:4]:
                try:
                    audit = getattr(self.reader, 'audit_sections', None)
                    if callable(audit):
                        local_records = audit(did, sections=('methods', 'results'), per_section=1)
                        section_audit.extend(local_records)
                        self.trace.emit({'type':'section_audit','document_id':did,
                                         'sections':[{ 'section':r['section'], 'found':r['found'] }
                                                     for r in local_records]})
                    else:
                        # Preserve the original SectionReader plugin contract.
                        local = self.reader.open(did, question, limit=2)
                        section_audit.extend({'document_id':s.document_id,'chunk_id':s.chunk_id,
                                              'page':s.page, 'local_open_only':True}
                                             for s in local)
                except (ValueError, OSError, sqlite3.Error) as exc:
                    self.trace.emit({'type':'open_error','document_id':did,'error_type':type(exc).__name__})
        answer=self.answerer.compose(question,sources,assessment,fallback_answer)
        result={'answer':answer,'assessment':assessment.status,
                'claim_facts_verified':False, 'required_ids':list(assessment.required_ids),
                'observed_ids':list(assessment.observed_ids), 'missing_ids':list(assessment.missing_ids),
                'unresolved_anchors':list(identity.unresolved), 'sources':[s.__dict__ for s in sources],
                'section_audit':section_audit, 'attempts':attempts,'http_calls':calls,
                'wall_ms':round((time.monotonic()-started)*1000,1),
                'note':'Structural study coverage != factual clinical verification.'}
        self.trace.emit({'type':'done','assessment':assessment.status,'http_calls':calls,
                         'wall_ms':result['wall_ms'],'missing_ids':result['missing_ids']})
        return result
