"""Offline tests of actual Agentic RAG tool selection and evidence-driven hops."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mhrag_gpu_chat_musa import llm
from mhrag_gpu_chat_musa.agent import AgenticRetriever, uncovered_comparison_entities


class FakeSearch:
    def __init__(self):
        self.calls = []

    def retrieve(self, question, mode, doc_ids=None, limit=8):
        self.calls.append((question, mode, doc_ids))
        rows = {
            'initial': [{'chunk_id':11, 'document_id':7, 'title':'trial.md',
                         'page':1, 'snippet':'Clozapine patient cohort.'}],
            'follow-up': [{'chunk_id':21, 'document_id':7, 'title':'trial.md',
                           'page':2, 'snippet':'Safety findings in the cohort.'}],
        }
        found = rows.get(question, [])
        return found, {'candidate_count':len(found), 'scored_pairs':len(found),
                       'retrieval_ms':2.0}


class ComparisonSearch(FakeSearch):
    def retrieve(self, question, mode, doc_ids=None, limit=8):
        self.calls.append((question, mode, doc_ids))
        rows = {
            'GASH vs BESS 研究阶段 最终疗效结果': [
                {'chunk_id': 101, 'document_id': 44, 'title': 'BESS-trial.pdf',
                 'page': 3, 'snippet': 'BESS is a completed phase 2 trial.'}],
            'GASH': [{'chunk_id': 102, 'document_id': 52, 'title': 'GASH-protocol.pdf',
                      'page': 1, 'snippet': 'GASH is a study protocol with no final results.'}],
        }
        found = rows.get(question, [])
        if doc_ids is not None:
            found = [row for row in found if row['document_id'] in doc_ids]
        return found[:limit], {'candidate_count': len(found), 'scored_pairs': len(found),
                               'retrieval_ms': 2.0}


class TightBudgetSearch(FakeSearch):
    def retrieve(self, question, mode, doc_ids=None, limit=8):
        self.calls.append((question, mode, doc_ids))
        rows = {
            'both studies': [
                {'chunk_id': 101, 'document_id': 44, 'title':'BESS.pdf',
                 'page':1, 'snippet':'BESS trial design'},
                {'chunk_id': 102, 'document_id': 44, 'title':'BESS.pdf',
                 'page':2, 'snippet':'BESS phase 2 trial'}],
            'GASH study': [
                {'chunk_id': 103, 'document_id': 52, 'title':'GASH.pdf',
                 'page':1, 'snippet':'GASH study protocol'},
                {'chunk_id': 104, 'document_id': 49, 'title':'other.pdf',
                 'page':1, 'snippet':'Related topic'}],
            'GASH outcome': [
                {'chunk_id': 105, 'document_id': 52, 'title':'GASH.pdf',
                 'page':2, 'snippet':'GASH study planned primary endpoint'}],
        }
        found = rows.get(question, [])
        if doc_ids is not None:
            found = [r for r in found if r['document_id'] in doc_ids]
        return found[:limit], {'candidate_count':len(found), 'scored_pairs':len(found),
                               'retrieval_ms':2.0}


class AgentTests(unittest.TestCase):
    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_second_hop_observes_first_evidence(self, decide, _configured):
        plans = [
            {'action':'search','tool':'semantic_search','query':'initial'},
            {'action':'search','tool':'document_search','query':'follow-up',
             'document_id':7, 'missing':'Safety endpoint'},
            {'action':'finish','reason':'Both facts supported'},
        ]
        observed = []
        def choose(question, sources, events, remaining, **kwargs):
            observed.append(len(sources))
            return plans.pop(0)
        decide.side_effect = choose
        core=FakeSearch()
        sources, trace=AgenticRetriever(core).run('Clozapine efficacy and safety',doc_ids=[7,8])
        self.assertEqual(observed,[0,1,2])
        self.assertEqual([s['citation'] for s in sources],[1,2])
        self.assertEqual(core.calls,[('initial','hybrid',[7,8]),('follow-up','hybrid_ce',[7])])
        self.assertEqual(trace['stop_reason'],'evidence_sufficient')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide',return_value={'action':'search','tool':'keyword_search','query':'initial'})
    def test_repeated_search_stops(self, _decision, _configured):
        core=FakeSearch()
        _, trace=AgenticRetriever(core,max_steps=5).run('question')
        self.assertEqual(len(core.calls),1)
        self.assertEqual(trace['stop_reason'],'repeated_search')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_repeated_query_recovers_missing_study(self, decide, _configured):
        question='比较GASH与BESS两项研究的研究阶段及是否有最终疗效结果。'
        repeated={'action':'search', 'tool':'reranked_search',
                  'query':'GASH vs BESS 研究阶段 最终疗效结果',
                  'missing':'GASH研究的具体阶段和最终疗效结果'}
        decide.side_effect=[repeated,repeated,{'action':'finish','reason':'Both covered'}]
        core=ComparisonSearch()
        evidence,trace=AgenticRetriever(core,max_steps=3,max_sources=4).run(question)
        self.assertEqual(core.calls,[
            ('GASH vs BESS 研究阶段 最终疗效结果','hybrid_ce',None),
            ('GASH','bm25',None)])
        self.assertEqual([s['document_id'] for s in evidence],[44,52])
        self.assertEqual(trace['steps'][1]['controller_override'],
                         'uncovered_comparison_entity')
        self.assertEqual(trace['uncovered_entities'],[])
        self.assertEqual(trace['stop_reason'],'evidence_sufficient')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_premature_finish_retrieves_uncovered_study(self, decide, _configured):
        question='比较GASH与BESS的疗效证据。'
        decide.side_effect=[
            {'action':'search','tool':'reranked_search',
             'query':'GASH vs BESS 研究阶段 最终疗效结果'},
            {'action':'finish','missing':'GASH研究缺少直接证据'},
            {'action':'finish','reason':'Done'}]
        core=ComparisonSearch()
        evidence,trace=AgenticRetriever(core,max_steps=3,max_sources=4).run(question)
        self.assertEqual(core.calls[-1],('GASH','bm25',None))
        self.assertEqual([s['document_id'] for s in evidence],[44,52])
        self.assertEqual(trace['stop_reason'],'evidence_sufficient')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_gap_retry_keeps_document_scope_and_reports_incomplete(self, decide, _configured):
        question='比较GASH与BESS两项研究的研究阶段及是否有最终疗效结果。'
        repeated={'action':'search','tool':'reranked_search',
                  'query':'GASH vs BESS 研究阶段 最终疗效结果',
                  'missing':'缺少GASH的研究阶段'}
        decide.side_effect=[repeated,repeated,{'action':'finish'}]
        core=ComparisonSearch()
        evidence,trace=AgenticRetriever(core,max_steps=2,max_sources=4).run(
            question,doc_ids=[44])
        self.assertEqual([c[2] for c in core.calls],[[44],[44]])
        self.assertEqual([s['document_id'] for s in evidence],[44])
        self.assertEqual(trace['uncovered_entities'],['GASH'])
        self.assertEqual(trace['stop_reason'],'coverage_gap')

    def test_comparison_acronym_detection_does_not_claim_factual_support(self):
        question='比较GASH与BESS两项研究。'
        only_bess=[{'title':'BESS.pdf','snippet':'BESS phase II'}]
        self.assertEqual(uncovered_comparison_entities(question, only_bess),['GASH'])
        self.assertEqual(uncovered_comparison_entities('BESS的终点是什么？', []),[])

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_generic_missing_evidence_rephrases_repeated_query(self, decide, _configured):
        repeated={'action':'search','tool':'semantic_search','query':'initial',
                  'missing':'safety endpoint study data'}
        decide.side_effect=[repeated,repeated,{'action':'finish'}]
        core=FakeSearch()
        _,trace=AgenticRetriever(core,max_steps=3).run('efficacy and safety')
        self.assertEqual(core.calls[:2], [
            ('initial','hybrid',None), ('safety endpoint study data','bm25',None)])
        self.assertEqual(trace['steps'][1]['controller_override'],'missing_evidence_query')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_tight_evidence_budget_reserves_final_hop(self, decide, _configured):
        """Regression: actual 4090 run filled four slots before third hop."""
        decide.side_effect = [
            {'action':'search', 'tool':'reranked_search','query':'both studies'},
            {'action':'search', 'tool':'reranked_search','query':'GASH study',
             'missing':'GASH结果是否已公布'},
            {'action':'search', 'tool':'document_search','query':'GASH outcome',
             'document_id':52, 'missing':'GASH终点结果数据'},
            {'action':'finish','missing':'GASH文章仅报告方案'}]
        core=TightBudgetSearch()
        sources,trace=AgenticRetriever(core,max_steps=3,max_sources=4).run(
            '比较GASH与BESS研究阶段和疗效结果')
        self.assertEqual(len(core.calls),3)
        self.assertEqual(core.calls[2],('GASH outcome','hybrid_ce',[52]))
        self.assertEqual([s['document_id'] for s in sources],[44,44,52,52])
        self.assertEqual(trace['evidence_count'],4)
        self.assertEqual(trace['stop_reason'],'evidence_budget')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_finish_with_explicit_gap_retrieves_more(self, decide, _configured):
        decide.side_effect = [
            {'action':'search','tool':'keyword_search','query':'initial'},
            {'action':'finish','missing':'Safety findings in the cohort'},
            {'action':'finish','missing':'无'}]
        core=FakeSearch()
        evidence,trace=AgenticRetriever(core,max_steps=3,max_sources=4).run(
            'Clozapine efficacy and safety')
        self.assertEqual(core.calls[1][0],'Safety findings in the cohort')
        self.assertEqual(trace['steps'][1]['controller_override'],
                         'missing_evidence_query')
        self.assertEqual(trace['stop_reason'],'evidence_sufficient')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_finish_does_not_mark_known_gap_as_sufficient(self, decide, _configured):
        decide.side_effect = [
            {'action':'search','tool':'keyword_search','query':'initial'},
            {'action':'finish','missing':'未提供最终疗效证据'},
            {'action':'finish','missing':'未提供最终疗效证据'},
            {'action':'finish','missing':'未提供最终疗效证据'}]
        core=FakeSearch()
        _,trace=AgenticRetriever(core,max_steps=4,max_sources=4).run('treatment results')
        self.assertEqual(trace['stop_reason'],'evidence_gap')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide',return_value={'action':'search','tool':'document_search',
                                                  'query':'initial','document_id':999})
    def test_unobserved_document_restricted(self, _decision, _configured):
        core=FakeSearch()
        AgenticRetriever(core,max_steps=1).run('question',doc_ids=[7])
        self.assertEqual(core.calls[0],('initial','hybrid_ce',[7]))

    @patch.object(llm,'configured',return_value=False)
    def test_model_required(self,_configured):
        with self.assertRaisesRegex(ValueError,'需要配置'):
            AgenticRetriever(FakeSearch()).run('question')

    @patch.object(llm,'call',return_value='```json\n{"action":"search","tool":"keyword_search","query":"HR"}\n```')
    def test_planner_json(self,_call):
        self.assertEqual(llm.agent_decide('HR?',[],[],2)['query'],'HR')

    @patch.object(llm,'call',return_value='以下是检索决策：\n{"action":"search","tool":"keyword_search","query":"BESS"}\n说明完毕')
    def test_planner_json_with_surrounding_text(self,_call):
        self.assertEqual(llm.agent_decide('BESS?',[],[],2)['query'],'BESS')

    @patch.object(llm,'call',side_effect=[
        '```json\n{"action":"search","tool":',
        '{"action":"search","tool":"keyword_search","query":"GASH"}'])
    def test_planner_invalid_json_retries_once(self,call_mock):
        self.assertEqual(llm.agent_decide('GASH?',[],[],2)['query'],'GASH')
        self.assertEqual(call_mock.call_count,2)

    @patch.object(llm,'call',return_value='{"action":"run_shell","query":"delete"}')
    def test_no_arbitrary_tool(self,_call):
        with self.assertRaises(llm.LLMError):llm.agent_decide('query',[],[],1)


if __name__ == '__main__':unittest.main()
