import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mhrag_v13.contracts import Resolution, Study, SearchResult, Source
from mhrag_v13.resolve import anchors, SQLiteStudyResolver
from mhrag_v13.routing import decide_route
from mhrag_v13.grounded_controller import GroundedController
from benchmarks.v13_pilot_runner import summary


class V07RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'knowledge.sqlite'
        with sqlite3.connect(self.db) as conn:
            conn.executescript('CREATE TABLE documents(id INTEGER PRIMARY KEY, filename TEXT);'
                               'CREATE TABLE chunks(id INTEGER PRIMARY KEY, document_id INTEGER, ordinal INTEGER, page TEXT, body TEXT);')
            conn.executemany('INSERT INTO documents VALUES (?,?)', [
                (44, 'BESS-trial.pdf'), (55, 'Focus-Pro-study.pdf'), (88, 'jcm-15-01379.pdf'),
                (101, 'ijms-17-01562.pdf')])
            conn.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)', [
                (1011, 44, 10, '2', 'Methods Study Design: A multicentre randomized controlled trial was conducted with infants requiring ventilation.'),
                (1012, 44, 22, '5', 'Results Trial recruitment included 232 participants who received randomized assigned interventions.'),
                (1013, 44, 24, '6', 'Geometric mean ratio was 1.02, confidence interval was 0.84 to 1.24, and p=0.86 in analysis.'),
                (2011, 55, 10, '2', 'Methods Study Design: A randomized controlled trial investigated treatment in children with ADHD.'),
                (2012, 55, 47, '7', 'Results Baseline Characteristics of enrolled participants were reported in the trial.'),
                (2013, 55, 52, '8', 'In the results group comparisons, p=0.006 and Cohen d=0.809 were reported.'),
                (3011, 88, 0, '1', 'This paper concerns a topic and reports findings in an original research report.'),
                (3012, 88, 1, '2', 'Primary outcome is described in this original report.'),
                (4011, 101, 0, '1', 'A review article and its introduction are presented here.'),
            ])

    def tearDown(self):
        self.temp.cleanup()

    def _api(self, doc, chunk):
        return {'document_id':doc,'chunk_id':chunk,'snippet':'Results: directly retrieved excerpt from local source.'}

    def test_topic_is_not_a_study(self):
        self.assertEqual(anchors('Focus Pro研究有没有证明数字疗法可以让所有ADHD儿童停止药物？'), ('Focus Pro',))

    def test_pdf_matches_filename_without_chinese_prefix(self):
        self.assertEqual(anchors('请找到名为 jcm-15-01379.pdf 的论文'), ('jcm-15-01379.pdf',))
        self.assertEqual(SQLiteStudyResolver(self.db).resolve('请找到名为 jcm-15-01379.pdf 的论文').ids, (88,))

    def test_quoted_title_not_fragmented(self):
        q='比较《Autoimmune Diseases: Molecular Pathogenesis and Therapeutic Targets》和《Update on systemic lupus erythematosus: A review》'
        self.assertEqual(anchors(q),('Autoimmune Diseases: Molecular Pathogenesis and Therapeutic Targets',
                                     'Update on systemic lupus erythematosus: A review'))

    def test_one_known_study_routes_to_partial_not_unverified_second(self):
        r=Resolution((Study(55,'Focus','Focus Pro'),),('unresolved_comparison_study',),('Focus Pro',))
        route=decide_route(r,[self._api(44,1011),self._api(44,1012)],self.db)
        self.assertEqual(route.mode,'partial_named')
        self.assertEqual(route.document_ids,(55,))
        self.assertFalse(route.provisional)

    def test_unknown_named_study_blocked(self):
        r=Resolution((),('Unknown Clinical Trial',),('Unknown Clinical Trial',))
        self.assertEqual(decide_route(r,[self._api(44,1011),self._api(44,1012)],self.db).mode, 'blocked_named')

    def test_no_name_single_chunk_not_promoted(self):
        self.assertEqual(decide_route(Resolution(),[self._api(44,1011)],self.db).mode,'blocked_no_anchor')

    def test_no_name_two_real_chunks_provisional(self):
        result=decide_route(Resolution(),[self._api(44,1011),self._api(44,1012)],self.db)
        self.assertEqual(result.mode,'provisional_api')
        self.assertEqual(result.document_ids,(44,))
        self.assertTrue(result.provisional)
        self.assertFalse(result.to_dict()['named_studies_complete'])

    def test_api_invalid_ids_not_provisional(self):
        self.assertEqual(decide_route(Resolution(),[self._api(44,987654),self._api(44,987655)],self.db).mode,'blocked_no_anchor')

    def test_two_source_docs_are_provisional_not_named(self):
        src=[self._api(44,1011),self._api(44,1012),self._api(55,2011),self._api(55,2012)]
        rt=decide_route(Resolution(),src,self.db)
        self.assertEqual(rt.document_ids,(44,55))
        self.assertEqual(rt.mode,'provisional_api')

    def test_unscoped_provisional_does_not_issue_full_answer(self):
        class Resolver:
            def resolve(self,question):return Resolution()
        class Retriever:
            def search(self,question,*,agentic=False,document_ids=None):
                return SearchResult([Source(44,1011,'2','BESS','Methods randomized trial'),
                                     Source(44,1012,'5','BESS','Results recruited participants')],
                                    'UNVERIFIED API ANSWER',7,{'planner_errors':0,'agentic':agentic})
        class Generator:
            def generate(self,question,spans,slots):
                s=next(x for x in spans if x.section=='methods')
                return {'claims':[{'study_id':44,'slot':'design',
                                   'evidence_id':s.evidence_id,
                                   'quote':s.text[:100],
                                   'text':'这是一项多中心随机对照试验。'}],
                        'generation_calls':1,'generation_status':'ok','diagnostics':[]}
        c=GroundedController(resolver=Resolver(),retriever=Retriever(),db_path=self.db,
                             generator=Generator())
        r=c.run('这项研究的设计是什么')
        self.assertEqual(r['routing']['mode'],'provisional_api')
        self.assertEqual(r['accepted_claim_count'],1)
        self.assertEqual(r['decision']['decision'],'partial_or_abstain')
        self.assertNotIn('UNVERIFIED API ANSWER',r['answer'])
        self.assertFalse(r['medical_claims_verified'])

    def test_explicit_unknown_does_not_run_model(self):
        class Resolver:
            def resolve(self,q):return Resolution((),('missing_study',),('missing_study',))
        class Retriever:
            def search(self,q,*,agentic=False,document_ids=None):
                return SearchResult([], '', 0,{'planner_errors':0,'agentic':agentic})
        class BadGenerator:
            def generate(self,*args):raise AssertionError('Must not run')
        c=GroundedController(resolver=Resolver(),retriever=Retriever(),db_path=self.db,
                             generator=BadGenerator())
        r=c.run('比较这项已命名但无法找到的研究的疗效')
        self.assertEqual(r['routing']['mode'],'blocked_named')
        self.assertEqual(r['generation_status'],'skipped')
        self.assertIsNone(r['generation_error'])

    def test_summary_exposes_route_reasons(self):
        rows=[{'case_id':'a','arm':'G','status':'ok','wall_ms':2,
               'accepted_claim_count':0,'accepted_study_ids':[],
               'generation_status':'skipped','generation_error':None,
               'routing':{'mode':'blocked_named','reason':'unresolved_explicit_study_name'},
               'decision':{'decision':'abstain'}},
              {'case_id':'a','arm':'F','status':'ok','wall_ms':1}]
        result=summary(rows,[{'case_id':'a','arm':'F'},{'case_id':'a','arm':'G'}])
        self.assertEqual(result['arms']['G']['routing_modes'],{'blocked_named':1})
        self.assertEqual(result['arms']['G']['routing_reasons'],{'unresolved_explicit_study_name':1})

if __name__=='__main__':unittest.main()

class DiagnosisV07Tests(unittest.TestCase):
    def test_legacy_diagnostic_without_route_and_new_route(self):
        from benchmarks.v13_pilot_diagnose_v07 import diagnose
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'results.jsonl'
            records=[
                {'case_id':'old','arm':'G','identity':{'complete':False,'unresolved':['ADHD']},
                 'result':{'evidence':{'local_spans':[1,2]}},'generation_status':'skipped'},
                {'case_id':'new','arm':'G','routing':{'mode':'partial_named','reason':'some_named_studies_unresolved'},
                 'identity':{'complete':False},'generation_status':'partial','accepted_claim_count':1},
                {'case_id':'f','arm':'F','generation_status':'na'},
            ]
            file.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf-8')
            out=diagnose(file)
            self.assertEqual(out['cases'],2)
            self.assertEqual(out['route_counts'],{'legacy_identity_block':1,'partial_named':1})
            self.assertEqual(out['rows'][1]['accepted'],1)
