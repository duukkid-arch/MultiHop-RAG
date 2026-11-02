import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from mhrag_v13.contracts import Resolution,Study,Source,SearchResult
from mhrag_v13.evidence_slots import SQLiteEvidenceHarvester,requested_slots,RESULTS_HEADER,EvidenceSpan,Slot
from mhrag_v13.grounded_answers import (ClaimSupportVerifier,QwenJsonClient,ClaimDraftGenerator,
                                         EvidenceStopController,parse_json_message,render_answer)
from mhrag_v13.grounded_controller import GroundedController

class FakeResolver:
    def resolve(self,question):
        return Resolution((Study(44,'bess','BESS'),Study(55,'focus','Focus Pro')),(),('BESS','Focus Pro'))

class FakeRetriever:
    def search(self,question,*,agentic=False,document_ids=None):
        return SearchResult([Source(44,101,'5','BESS','Results From Dec 18, 2018, 232 infants were randomized'),
                             Source(55,202,'6','Focus Pro','Results TOVA p=0.006 Cohen d=0.809')],
                            'OLD ANSWER (do not keep)',12,{'planner_errors':0,'agentic':agentic})

class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.db=Path(self.t.name)/'fake.sqlite'
        with sqlite3.connect(self.db) as c:
            c.executescript('CREATE TABLE documents (id integer primary key, filename text);'
                            'CREATE TABLE chunks (id integer primary key, document_id integer, ordinal integer, page text, body text);')
            c.executemany('INSERT INTO documents VALUES (?,?)',[(44,'bess.pdf'),(55,'focus.pdf')])
            rows=[
                (101,44,12,'2','Methods Study design: Infants diagnosed with bronchiolitis were included. Trial design and controls.'),
                (102,44,14,'3','Primary outcome was total duration of mechanical ventilation, recorded in hours.'),
                (103,44,52,'5','Results From Dec 18, 2018, 232 infants were randomized into two groups.'),
                (104,44,62,'6','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86) in the analysis.'),
                (105,44,87,'9','findings may not be generalized to all study populations. This is discussion.'),
                (201,55,3,'3','Methods Study design: A prospective single-center randomized controlled study of pediatric ADHD.'),
                (202,55,4,'1','Results: The TOVA attention measure improved (p=0.006, Cohen d=0.809) in the treated group.'),
                (203,55,30,'4','Primary outcome included objective neurocognitive attention scores and TOVA.'),
                (204,55,47,'7','Results Baseline Characteristics. Randomization and intention to treat analysis were followed.'),
                (205,55,53,'7','Between-group objective measures had p=0.006 and Cohen d=0.809.'),
                (206,55,91,'12','findings may not be generalized to the broader population. This is discussion.'),
            ]
            c.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',rows)
        self.h=SQLiteEvidenceHarvester(self.db)

    def tearDown(self):self.t.cleanup()

    def test_headings_filter_false_sentence(self):
        self.assertIsNone(RESULTS_HEADER.search('findings may not be generalized.'))
        self.assertIsNotNone(RESULTS_HEADER.search('some paragraph\nResults From 2024, we observed outcomes.'))
        self.assertIsNotNone(RESULTS_HEADER.search('Results Baseline Characteristics'))
    def test_sqlite_read_only(self):
        spans,slots=self.h.harvest([44,55],'比较研究设计、主要终点和结果')
        self.assertTrue(spans)
        self.assertEqual({44,55},{s.document_id for s in spans})
        self.assertFalse(any(s.ordinal in (87,91) for s in spans))
        self.assertTrue(any(s.ordinal==62 for s in spans))
        self.assertTrue(any(s.ordinal==53 for s in spans))
        self.assertTrue(all(s.origin=='sqlite_local_open' for s in spans))
        self.assertTrue(any(s.slot=='results' and s.status=='candidate' for s in slots))
    def test_invalid_doc_id(self):
        with self.assertRaises(ValueError):self.h.harvest([0],'hello')
    def test_question_slots(self):
        self.assertEqual(requested_slots('比较研究设计、主要终点和结果'),('design','primary_outcome','results'))
    def test_quote_pass(self):
        s=EvidenceSpan('E44_104',44,104,62,'6','results','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':44,'slot':'results','text':'几何均值比 1.02，p=0.86。',
           'quote':'Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).',
           'evidence_id':'E44_104'}
        v=ClaimSupportVerifier().review([c],[s])
        self.assertEqual(len(v['accepted']),1)
        self.assertFalse(v['medical_claims_verified'])
    def test_wrong_doc_rejected(self):
        s=EvidenceSpan('E44_104',44,104,62,'6','results','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':55,'slot':'results','text':'几何均值比 1.02，p=0.86。',
           'quote':s.text,'evidence_id':s.evidence_id}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],'wrong_study_id')
    def test_wrong_numeric_rejected(self):
        s=EvidenceSpan('E44_104',44,104,62,'6','results','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':44,'slot':'results','text':'几何均值比 1.80，p=0.86。',
           'quote':s.text,'evidence_id':s.evidence_id}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],'unsupported_numeric_in_claim')
    def test_fake_quote_rejected(self):
        s=EvidenceSpan('E44_104',44,104,62,'6','results','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':44,'slot':'results','text':'几何均值比为 1.02。',
           'quote':'Completely fabricated ratio was 1.02 in the intervention group.',
           'evidence_id':s.evidence_id}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],'quote_not_in_original_chunk')
    def test_results_using_methods_rejected(self):
        s=EvidenceSpan('E44_101',44,101,12,'2','methods','Methods Trial design and expected treatment results were described')
        c={'study_id':44,'slot':'results','text':'该项研究描述了预期治疗结果。',
           'quote':s.text,'evidence_id':s.evidence_id}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],'non_results_section')
    def test_plain_text_json(self):
        self.assertEqual(parse_json_message('```json\n{"claims": []}\n```'),{'claims':[]})
    def test_bad_json(self):
        with self.assertRaises(ValueError):parse_json_message('no json')
    def test_judge_unsupported(self):
        s=EvidenceSpan('E44_104',44,104,62,'6','results','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':44,'slot':'results','text':'几何均值比为 1.02。','quote':s.text,'evidence_id':s.evidence_id}
        client=QwenJsonClient(completion=lambda system,user:json.dumps({'judgments':[{'claim_index':0,'verdict':'unsupported'}]}))
        v=ClaimSupportVerifier(client).review([c],[s])
        self.assertEqual(len(v['accepted']),0)
    def test_judge_failure_fail_closed(self):
        s=EvidenceSpan('E44_104',44,104,62,'6','results','Geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':44,'slot':'results','text':'几何均值比为 1.02。','quote':s.text,'evidence_id':s.evidence_id}
        client=QwenJsonClient(completion=lambda s,u:'not json')
        self.assertEqual(len(ClaimSupportVerifier(client).review([c],[s])['accepted']),0)
    def test_stop_missing(self):
        ident=FakeResolver().resolve('x')
        out=EvidenceStopController().decide(ident,[Slot(44,'results','candidate',('E44_104',),'')],{'accepted':[]},('results',))
        self.assertEqual(out['decision'],'partial_or_abstain')
        self.assertIn({'document_id':55,'slot':'results'},out['missing_slots'])
    def test_pipeline_isolation_and_no_reuse_old_answer(self):
        client=QwenJsonClient(completion=lambda s,u:json.dumps({'claims':[], 'limitations':['不足']}))
        ctrl=GroundedController(resolver=FakeResolver(),retriever=FakeRetriever(),db_path=self.db,
                                generator=ClaimDraftGenerator(client),claim_verifier=ClaimSupportVerifier())
        result=ctrl.run('比较研究设计、主要终点和结果')
        self.assertNotIn('OLD ANSWER',result['answer'])
        self.assertFalse(result['medical_claims_verified'])
        self.assertTrue(result['human_medical_review_required'])
        self.assertIn('api_sources',result['evidence'])
        self.assertIn('local_spans',result['evidence'])
        self.assertEqual(result['retrieval']['http_calls'],1)
    def test_evidence_only_no_model(self):
        class ForbiddenGenerator:
            def generate(self,*a):raise AssertionError('must not call model')
        ctrl=GroundedController(resolver=FakeResolver(),retriever=FakeRetriever(),db_path=self.db,
                                generator=ForbiddenGenerator())
        result=ctrl.run('比较研究结果',evidence_only=True)
        self.assertEqual(result['decision']['decision'],'evidence_audit_only')
        self.assertEqual(result['draft_claim_count'],0)
    def test_unknown_study_abstain(self):
        ident=Resolution((),('unknown acronym',),('unknown acronym',))
        out=EvidenceStopController().decide(ident,[],{'accepted':[]},('results',))
        self.assertEqual(out['decision'],'abstain')
    def test_genuine_source_ids(self):
        spans,_=self.h.harvest([44,55],'研究结果')
        self.assertEqual(len({s.evidence_id for s in spans}),len(spans))
        self.assertTrue(all(s.evidence_id == f'E{s.document_id}_{s.chunk_id}' for s in spans))

if __name__=='__main__':unittest.main()
