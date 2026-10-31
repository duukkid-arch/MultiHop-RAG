"""v12.4 integrated regressions (real SQLite store, fake 1024D encoder, no GPU)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mhrag_gpu_chat_musa.core import ChatCore, named_study_terms
from mhrag_gpu_chat_musa import llm
from mhrag_gpu_chat_musa.agent import AgenticRetriever, named_comparison_terms


class FakeEncoder:
    device = 'cpu'
    embedder = 'fake'
    reranker = 'fake'

    def embed(self, texts):
        arr = np.zeros((len(texts), 1024), dtype=np.float32)
        arr[:, 0] = 1.0
        return arr

    def score(self, pairs):
        # Deliberately reward irrelevant articles to verify strict target scope.
        return [9.0 if 'pylori' in x[1].lower() else 0.3 for x in pairs]


class IntegratedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.core = ChatCore(Path(self.tmp.name) / 'test.sqlite', FakeEncoder())
        self.bess = self.core.ingest('bess_trial.txt',
            ('BESS phase II trial main endpoint ventilation time. '
             'BESS primary outcome duration mechanical ventilation. '*7).encode())['id']
        self.focus = self.core.ingest('doc55.txt',
            ('Focus Pro randomized trial comparative efficacy study. '
             'Focus Pro intervention standard control group and outcomes. '*7).encode())['id']
        self.gash = self.core.ingest('gash_trial.txt',
            ('GASH protocol subarachnoid hemorrhage glibenclamide. '
             'GASH planned mRS endpoint, no final efficacy results. '*7).encode())['id']
        self.noise = self.core.ingest('unrelated.txt',
            ('H pylori trial primary endpoint breath test treatment. '*9).encode())['id']

    def tearDown(self):
        self.tmp.cleanup()

    def test_mixed_case_focus_pro(self):
        self.assertEqual(named_study_terms('比较 Focus Pro 与 BESS 两项研究'),
                         ['BESS', 'Focus Pro'])
        self.assertIn('NeuroAI',named_comparison_terms('两篇NeuroAI观点文献'))

    def test_comparison_scope_union(self):
        found,trace=self.core.agent_tool_search('outcomes','hybrid_ce',
                  original_question='比较 Focus Pro 与 BESS 两项研究')
        self.assertEqual(trace['entity_scope'],[self.bess,self.focus])
        self.assertTrue(found)
        self.assertTrue(set(x['document_id'] for x in found) <= {self.bess,self.focus})
        self.assertTrue(trace['original_query_fused'])

    def test_missing_study_follow_up_scopes_to_just_that_study(self):
        found,trace=self.core.agent_tool_search('Focus Pro','hybrid_ce',
                  original_question='比较 Focus Pro 与 BESS 两项研究')
        self.assertEqual(trace['entity_scope'],[self.focus])
        self.assertTrue(found)
        self.assertEqual(set(x['document_id'] for x in found),{self.focus})

    def test_explicit_user_scope_overrules_named_study(self):
        found,trace=self.core.agent_tool_search('BESS','hybrid_ce',
                original_question='BESS研究主要终点',doc_ids=[self.noise])
        self.assertEqual(found,[])
        self.assertEqual(trace['entity_scope'],[])

    def test_unknown_study_is_not_hard_scoped(self):
        _,trace=self.core.agent_tool_search('XZZZ study','bm25',
             original_question='XZZZ study design')
        self.assertIsNone(trace['entity_scope'])

    @patch.object(llm, 'configured', return_value=True)
    @patch.object(llm, 'agent_decide')
    def test_actual_controller_uses_core_agent_tool_search(self,decide,_):
        decide.side_effect=[
            {'action':'search','tool':'reranked_search','query':'BESS trial'},
            {'action':'search','tool':'reranked_search','query':'similar effects',
             'missing':'Focus Pro'},
            {'action':'finish','missing':'无'}]
        found,trace=AgenticRetriever(self.core,max_steps=3,max_sources=4).run(
                         '比较 Focus Pro 与 BESS 两项研究')
        docs={x['document_id'] for x in found}
        self.assertTrue({self.bess,self.focus} <= docs)
        steps=[s for s in trace['steps'] if s['action']=='search']
        self.assertTrue(steps[1]['controller_override']=='uncovered_comparison_entity')
        self.assertEqual(steps[1]['entity_scope'],[self.focus])

    @patch.object(llm, 'configured', return_value=True)
    @patch.object(llm, 'agent_decide')
    def test_json_planner_failure_recovers_into_targeted_search(self,decide,_):
        decide.side_effect=[
            {'action':'search','tool':'keyword_search','query':'BESS'},
            llm.LLMError('bad json'),
            {'action':'finish','missing':'无'}]
        found,trace=AgenticRetriever(self.core,max_steps=3,max_sources=4).run(
                         '比较GASH与BESS两项试验')
        self.assertEqual(trace['planner_failures'],1)
        self.assertTrue({self.bess,self.gash} <= {x['document_id'] for x in found})

    @patch.object(llm, 'call', side_effect=[
        'prefix {"action":"search", "tool":',
        '{"action":"search","tool":"keyword_search","query":"GASH"}'])
    def test_invalid_json_repairs_once(self,client):
        self.assertEqual(llm.agent_decide('GASH?',[],[],2)['query'],'GASH')
        self.assertEqual(client.call_count,2)

    @patch.object(llm, 'call', return_value='{"action":"run_shell","query":"rm -rf"}')
    def test_unauthorized_tool_never_retried(self,client):
        with self.assertRaises(llm.LLMError):
            llm.agent_decide('BESS?',[],[],2)
        self.assertEqual(client.call_count,1)

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    @patch.object(llm,'answer',return_value='文献资料显示这一研究有明确的设计与主要终点[1]。')
    def test_real_chat_integration(self,answer,decide,_):
        decide.side_effect=[
            {'action':'search','tool':'keyword_search','query':'BESS'},
            {'action':'finish','missing':'无'}]
        result=self.core.chat('BESS试验研究设计是什么?',mode='bm25',agentic=True)
        self.assertTrue(result['sources'])
        self.assertTrue(result['agentic'])
        self.assertIn('steps',result['trace'])


if __name__=='__main__':unittest.main()

class ProvenanceTests(unittest.TestCase):
    def test_mentioned_study_is_not_same_as_retrieved_study(self):
        from mhrag_gpu_chat_musa.agent import uncovered_comparison_entities
        sources=[{'document_id':44,'title':'BESS_trial.pdf',
                  'snippet':'BESS trial cites Focus Pro as related prior work.'}]
        # Textual mention alone incorrectly looks complete; a document map
        # prevents a cited study from masquerading as the source paper.
        self.assertEqual(uncovered_comparison_entities('比较 Focus Pro 与 BESS',sources,
                         {'Focus Pro':{55},'BESS':{44}}),['Focus Pro'])
