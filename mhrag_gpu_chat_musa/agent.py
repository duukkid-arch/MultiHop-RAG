"""Bounded Agentic RAG: plan -> tool -> observe -> re-plan -> answer.

Tools are read-only retrievers over the selected local document collection.
The LLM can choose a new query after inspecting actual evidence, including
follow-up searches scoped to a document it discovered in a previous hop.
"""
from __future__ import annotations

import math
import os
import re
import time

from . import llm

TOOLS = {'keyword_search': 'bm25', 'semantic_search': 'hybrid',
         'reranked_search': 'hybrid_ce', 'document_search': 'hybrid_ce'}
MODE_TOOL = {'bm25': 'keyword_search', 'hybrid': 'semantic_search',
             'hybrid_ce': 'reranked_search'}


def has_evidence_gap(missing):
    gap = str(missing or '').strip()
    return bool(gap and gap.casefold() not in (
        '无', 'none', '无缺口', '没有缺口', '无明显缺口', '无证据缺口',
        'no missing evidence', 'no gaps', 'n/a'))


def named_comparison_terms(question):
    """Conservative acronym + mixed-case paper name extraction (retrieval hints only).

    Recognizes GASH/BESS, NeuroAI, and Focus Pro without hard-coding the
    benchmark's gold document IDs. Ignore generic field/metric abbreviations.
    """
    q = str(question)
    acronyms = re.findall(r'(?<![A-Za-z0-9])([A-Z][A-Z0-9-]{2,})(?![A-Za-z0-9])', q)
    mixed = re.findall(r'(?<![A-Za-z])([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?![A-Za-z])', q)
    camel = re.findall(r'(?<![A-Za-z])([A-Z][a-z]+[A-Z][A-Za-z]+)(?![A-Za-z])', q)
    ignore = {'PDF','RAG','LLM','BMI','CI','HR','MRI','CT','RCT','USA','ACS',
              'BM25','TOP','PICO','AI','GPT','COVID','MDD'}
    terms=[]
    for x in [*acronyms,*mixed,*camel]:
        if x.upper() not in ignore and x.casefold() not in [y.casefold() for y in terms]:
            terms.append(x)
    return terms[:5]


def uncovered_comparison_entities(question, sources, expected_docs=None):
    """Conservative coverage hint for questions comparing named acronyms.

    Requiring at least two distinct acronyms avoids imposing this heuristic on
    ordinary single-topic questions. Presence is only a retrieval signal, not
    proof that an article supports any particular clinical conclusion.
    """
    entities = named_comparison_terms(question)
    if len(entities) < 2:
        return []
    observed = ' '.join(str(s.get('title', '')) + ' ' + str(s.get('snippet', ''))
                        for s in sources).casefold()
    missing = []
    for entity in entities:
        # Document provenance beats a stray study name quoted in references.
        ids = (expected_docs or {}).get(entity)
        if ids:
            if not any(s.get('document_id') in ids for s in sources):
                missing.append(entity)
        elif entity.casefold() not in observed:
            missing.append(entity)
    return missing


def coverage_retry(question, sources, missing, tried, allow_gap_query=True, expected_docs=None):
    """Choose a *new* read-only retrieval when the planner stalls on a gap."""
    uncovered = uncovered_comparison_entities(question, sources, expected_docs)
    # Prefer the specific study the model identifies as missing.
    targets = sorted(uncovered, key=lambda term: (term.casefold() not in missing.casefold(),
                                                  uncovered.index(term)))
    for entity in targets:
        for tool in ('keyword_search', 'semantic_search', 'reranked_search'):
            if normalized_signature(entity, tool, None) not in tried and not any(
                    t[0] == normalized_signature(entity, tool, None)[0] and t[1] == tool
                    for t in tried):
                return {'action': 'search', 'tool': tool, 'query': entity,
                        'reason': f'控制器纠偏：单独检索尚未覆盖的研究 {entity}',
                        'missing': missing or f'{entity} 缺少直接文献证据',
                        'controller_override': 'uncovered_comparison_entity'}
    # Other multi-hop questions may have no uppercase study names. A genuine
    # missing-evidence description is still preferable to repeating a failed
    # broad query. Never create a new search from an empty/no-gap assessment.
    gap = str(missing).strip()[:180]
    if allow_gap_query and sources and has_evidence_gap(gap):
        for tool in ('keyword_search', 'semantic_search'):
            if not any(t[0] == normalized_signature(gap, tool, None)[0] and t[1] == tool
                       for t in tried):
                return {'action': 'search', 'tool': tool, 'query': gap,
                        'reason': '控制器纠偏：根据尚未解决的证据缺口重新检索',
                        'missing': gap, 'controller_override': 'missing_evidence_query'}
    return None


def fallback_decision(question, evidence, gaps, tried, default_tool, step, expected_docs=None):
    """Read-only deterministic fallback; never fabricate a sufficient-evidence verdict."""
    rescue = coverage_retry(question, evidence, gaps, tried, allow_gap_query=True,
                            expected_docs=expected_docs)
    if rescue is not None:
        return rescue
    if not evidence and not any(t[0] == normalized_signature(question, default_tool, None)[0] for t in tried):
        return {'action': 'search', 'tool': default_tool, 'query': question,
                'reason': '规划失败：使用默认检索器',
                'controller_override': 'planner_fallback_initial'}
    return None


def normalized_signature(query, tool, scope):
    return (' '.join(str(query).casefold().split()), tool,
            tuple(sorted(int(x) for x in scope)) if scope is not None else ())


def diagnostic_coverage(question, sources, expected_docs=None):
    """Heuristic retrieval diagnostics, NOT clinical claim verification."""
    uncovered = uncovered_comparison_entities(question, sources, expected_docs)
    doc_ids = sorted({str(s.get('document_id')) for s in sources
                      if s.get('document_id') is not None})
    return {'uncovered_entities': uncovered, 'observed_document_ids': doc_ids,
            'distinct_documents': len(doc_ids),
            'status': 'incomplete' if uncovered else 'unverified'}


def budget(name, default, minimum, maximum):
    try:
        n = int(os.getenv(name, str(default)))
    except ValueError:
        n = default
    return max(minimum, min(n, maximum))


class AgenticRetriever:
    def __init__(self, core, max_steps=None, max_sources=None):
        self.core = core
        self.max_steps = max_steps if max_steps is not None else budget('CHAT_AGENT_MAX_STEPS', 3, 1, 5)
        self.max_sources = max_sources if max_sources is not None else budget('CHAT_AGENT_MAX_SOURCES', 8, 2, 12)
        if not (1 <= self.max_steps <= 5 and 2 <= self.max_sources <= 12):
            raise ValueError('Agent 检索预算超出允许范围')

    def run(self, question, mode='hybrid_ce', doc_ids=None, history=None):
        if mode not in MODE_TOOL:
            raise ValueError('检索模式不支持')
        if not llm.configured():
            raise ValueError('Agentic RAG 需要配置 LLM 模型（可使用本地 API）')
        start = time.perf_counter()
        evidence, steps, seen, tried = [], [], set(), set()
        stop_reason, gaps = 'step_budget', ''
        planner_failures = 0
        no_gain_streak = 0
        study_doc_map = {}
        per_hop = max(2, math.ceil(self.max_sources / self.max_steps))
        agent_search = getattr(self.core, 'agent_tool_search', None)
        for i in range(self.max_steps):
            try:
                decision = llm.agent_decide(question, evidence, steps,
                                            self.max_steps-i,
                                            default_tool=MODE_TOOL[mode], history=history)
            except llm.LLMError as exc:
                planner_failures += 1
                decision = fallback_decision(question, evidence, gaps, tried,
                                             MODE_TOOL[mode], i, study_doc_map)
                steps.append({'step': i+1, 'action': 'planner_recovery',
                              'reason': type(exc).__name__,
                              'fallback_available': decision is not None})
                if decision is None:
                    stop_reason = 'planner_error' if not evidence else 'planner_recovery_exhausted'
                    break
            gaps = str(decision.get('missing', ''))[:220]
            # At hop >= 2, correct generic planner searches that ignore a named
            # comparison arm.  A missing study must get its own query before
            # the final step instead of repeatedly searching the broad question.
            if (i > 0 and decision['action'] == 'search' and evidence
                    and uncovered_comparison_entities(question, evidence, study_doc_map)):
                uncovered = uncovered_comparison_entities(question, evidence, study_doc_map)
                qnorm = str(decision.get('query', '')).casefold()
                if not any(term.casefold() in qnorm for term in uncovered):
                    directed = coverage_retry(question, evidence, gaps, tried,
                                              allow_gap_query=False,
                                              expected_docs=study_doc_map)
                    if directed is not None:
                        decision = directed
                        gaps = str(decision.get('missing', gaps))[:220]
            if decision['action'] == 'finish':
                if not evidence:
                    decision = {'action':'search', 'tool': MODE_TOOL[mode],
                                'query':question, 'reason':'还没有证据，必须先检索'}
                else:
                    # A planner saying "finish" while also identifying a
                    # concrete missing fact is not evidence sufficiency.
                    rescue = coverage_retry(question, evidence, gaps, tried,
                                            allow_gap_query=True,
                                            expected_docs=study_doc_map)
                    if rescue is None:
                        stop_reason = ('coverage_gap' if uncovered_comparison_entities(question, evidence, study_doc_map)
                                       else 'evidence_gap' if has_evidence_gap(gaps)
                                       else 'evidence_sufficient')
                        steps.append({'step':i+1, 'action':'finish',
                                      'reason':str(decision.get('reason',''))[:160], 'missing':gaps})
                        break
                    decision = rescue
            tool = decision.get('tool', MODE_TOOL[mode])
            if tool not in TOOLS:
                tool = MODE_TOOL[mode]
            query = str(decision.get('query', '')).strip()[:220] or question[:220]
            scope = None
            if tool == 'document_search':
                target = decision.get('document_id')
                observed = {s['document_id'] for s in evidence}
                if type(target) is int and target in observed:
                    scope = [target]
                else:
                    tool = MODE_TOOL[mode]
            if doc_ids is not None:
                allowed = {int(x) for x in doc_ids}
                scope = sorted(allowed & set(scope)) if scope is not None else sorted(allowed)
                if not scope:
                    stop_reason = 'document_scope_empty'
                    steps.append({'step': i+1, 'action':'stop', 'reason':'所选文档范围为空'})
                    break
            signature = normalized_signature(query, tool, scope)
            if signature in tried:
                rescue = coverage_retry(question, evidence, gaps, tried,
                                        expected_docs=study_doc_map)
                if rescue is None:
                    stop_reason = ('coverage_gap' if uncovered_comparison_entities(question, evidence, study_doc_map)
                                   else 'evidence_gap' if has_evidence_gap(gaps)
                                   else 'repeated_search')
                    steps.append({'step':i+1, 'action':'stop', 'reason':'重复检索已被阻止'})
                    break
                decision = rescue
                tool = decision['tool']
                query = decision['query']
                scope = None
                if doc_ids is not None:
                    scope = sorted(int(x) for x in doc_ids)
                signature = normalized_signature(query, tool, scope)
                if signature in tried:
                    stop_reason = 'repeated_search'
                    break
            gaps = str(decision.get('missing', gaps))[:220]
            tried.add(signature)
            if agent_search is not None:
                found, trace = agent_search(query, TOOLS[tool], doc_ids=scope,
                                            limit=per_hop+len(seen),
                                            original_question=question)
            else:
                # Lightweight offline cores still expose the original basic
                # retrieval interface; no named-entity hints are needed there.
                found, trace = self.core.retrieve(query, TOOLS[tool], doc_ids=scope,
                                                  limit=per_hop+len(seen))
            for name, ids in (trace.get('entity_document_map') or {}).items():
                if ids and name in named_comparison_terms(question):
                    study_doc_map[name] = set(ids)
            added = []
            # With a tight citation budget (e.g. four excerpts), greedy first
            # hops otherwise fill every slot before the model can inspect the
            # evidence and search for missing results in its last hop.
            # Reserve one slot until the final permitted search. A genuinely
            # sufficient answer can still finish earlier.
            reserve = 1 if i < self.max_steps - 1 and self.max_steps > 1 else 0
            evidence_cap = self.max_sources - reserve
            # Prefer a previously unseen paper over the third/second excerpt
            # of an already covered one; this helps two-paper comparisons with
            # a four-citation evidence budget without inventing evidence.
            if len(named_comparison_terms(question)) >= 2:
                observed_docs = {s['document_id'] for s in evidence}
                found = sorted(found, key=lambda s:(s['document_id'] in observed_docs,))
            for source in found:
                if source['chunk_id'] in seen:
                    continue
                if len(evidence) >= evidence_cap or len(added) >= per_hop:
                    break
                item = dict(source)
                item['citation'] = len(evidence)+1
                evidence.append(item)
                seen.add(item['chunk_id'])
                added.append(item['citation'])
            steps.append({'step':i+1, 'action':'search', 'tool':tool, 'query':query,
                          'document_id':scope[0] if tool=='document_search' else None,
                          'reason':str(decision.get('reason',''))[:160], 'missing':gaps,
                          'controller_override':decision.get('controller_override'),
                          'entity_scope':trace.get('entity_scope'),
                          'entity_names':trace.get('entity_names',[]),
                          'original_query_fused':trace.get('original_query_fused',False),
                          'hits':len(found), 'new_citations':added,
                          'candidate_count':trace.get('candidate_count',0),
                          'scored_pairs':trace.get('scored_pairs',0),
                          'retrieval_ms':trace.get('retrieval_ms',0)})
            no_gain_streak = 0 if added else no_gain_streak + 1
            if not added and no_gain_streak >= budget('CHAT_AGENT_NO_GAIN_LIMIT', 3, 1, 4):
                # Stop a useless search loop; report unresolved coverage separately.
                stop_reason = 'no_gain_budget'
                break
            if len(evidence) >= self.max_sources:
                stop_reason = 'evidence_budget'
                break

        if not evidence and stop_reason == 'step_budget':
            stop_reason = 'no_evidence'
        # Review the final fetched evidence without launching another retrieval.
        review_wants_search = False
        if evidence and stop_reason in ('step_budget', 'evidence_budget'):
            try:
                review = llm.agent_decide(question, evidence, steps, 0,
                                          default_tool=MODE_TOOL[mode], history=history)
                gaps = str(review.get('missing',''))[:220]
                steps.append({'step':len(steps)+1, 'action':'review',
                              'assessment':review['action'],
                              'reason':str(review.get('reason',''))[:160], 'missing':gaps})
                review_wants_search = review['action'] == 'search'
                if (review['action']=='finish' and stop_reason=='step_budget'
                        and not has_evidence_gap(gaps)):
                    stop_reason = 'evidence_sufficient'
            except llm.LLMError:
                steps.append({'step':len(steps)+1, 'action':'review',
                              'reason':'最终证据评估不可用'})
        coverage = diagnostic_coverage(question, evidence, study_doc_map)
        uncovered = coverage['uncovered_entities']
        if uncovered and stop_reason == 'evidence_sufficient':
            stop_reason = 'coverage_gap'
        return evidence, {
            'agentic':True, 'steps':steps, 'stop_reason':stop_reason,
            'evidence_gaps':gaps, 'uncovered_entities':uncovered,
            'coverage_diagnostics': coverage,
            'study_document_map': {k:sorted(v) for k,v in study_doc_map.items()},
            'planner_failures': planner_failures,
            'no_gain_streak': no_gain_streak,
            'evidence_count':len(evidence), 'max_sources':self.max_sources,
            'max_steps':self.max_steps, 'review_requests_more_evidence':review_wants_search,
            'queries':[s['query'] for s in steps if s['action']=='search'],
            'candidate_count':sum(s.get('candidate_count',0) for s in steps),
            'scored_pairs':sum(s.get('scored_pairs',0) for s in steps),
            'retrieval_ms':round(sum(s.get('retrieval_ms',0) for s in steps),1),
            'agent_ms':round((time.perf_counter()-start)*1000,1),
        }
