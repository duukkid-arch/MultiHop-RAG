"""Regression checks for safe controller improvements (no GPU needed)."""
import unittest
from unittest.mock import patch
from mhrag_gpu_chat_musa import llm
from mhrag_gpu_chat_musa.agent import (AgenticRetriever, normalized_signature,
                                        diagnostic_coverage)


class Core:
    def __init__(self): self.calls=[]
    def retrieve(self, query, mode, doc_ids=None, limit=8):
        self.calls.append((query, mode, doc_ids))
        rows = ({'start': [{'chunk_id':1,'document_id':44,'title':'BESS', 'snippet':'BESS trial'}],
                 'GASH': [{'chunk_id':2,'document_id':52,'title':'GASH','snippet':'GASH protocol'}]}
                .get(query, []))
        return [r for r in rows if doc_ids is None or r['document_id'] in doc_ids], {}


class OptimizationTests(unittest.TestCase):
    def test_signature_normalization_and_scope(self):
        self.assertEqual(normalized_signature(' ABC   def ', 'keyword_search',[9,2]),
                         normalized_signature('abc def','keyword_search',[2,9]))
        self.assertNotEqual(normalized_signature('abc','keyword_search',[2]),
                            normalized_signature('abc','keyword_search',[9]))

    def test_coverage_is_diagnostic_not_factual_verification(self):
        val = diagnostic_coverage('比较GASH和BESS',
                                  [{'document_id':44,'title':'BESS','snippet':'BESS'}])
        self.assertEqual(val['uncovered_entities'],['GASH'])
        self.assertEqual(val['status'],'incomplete')

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_second_hop_planner_failure_recovers_missing_entity(self,decide,_):
        decide.side_effect=[{'action':'search','tool':'keyword_search','query':'start',
                             'missing':'GASH study'}, llm.LLMError('bad json'),
                            {'action':'finish','reason':'done'}]
        core=Core()
        evidence,trace=AgenticRetriever(core,max_steps=3,max_sources=4).run('比较GASH和BESS')
        self.assertEqual([r['document_id'] for r in evidence],[44,52])
        self.assertEqual(trace['planner_failures'],1)
        self.assertEqual(trace['stop_reason'],'evidence_sufficient')

    @patch.dict('os.environ',{'CHAT_AGENT_NO_GAIN_LIMIT':'1'})
    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide',return_value={
        'action':'search','tool':'keyword_search','query':'nothing'})
    def test_no_gain_limit_from_environment(self,_decide,_configured):
        core=Core()
        _evidence,trace=AgenticRetriever(core,max_steps=4).run('question')
        self.assertEqual(trace['stop_reason'],'no_gain_budget')
        self.assertEqual(len(core.calls),1)

    @patch.object(llm,'configured',return_value=True)
    @patch.object(llm,'agent_decide')
    def test_planner_failure_with_evidence_terminates_conservatively(self,decide,_):
        decide.side_effect=[{'action':'search','tool':'keyword_search','query':'start'},
                            llm.LLMError('invalid')]
        core=Core()
        _,trace=AgenticRetriever(core,max_steps=3).run('single topic')
        self.assertEqual(trace['stop_reason'],'planner_recovery_exhausted')
        self.assertEqual(trace['planner_failures'],1)

if __name__=='__main__': unittest.main()
