"""v0.8 regression: intent-aware evidence for non-RCT papers without gold IDs."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from mhrag_v13.evidence_slots import requested_slots, SQLiteEvidenceHarvester, EvidenceSpan, Slot
from mhrag_v13.grounded_answers import ClaimDraftGenerator, ClaimSupportVerifier, QwenJsonClient


class IntentReaderV08(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/'corpus.sqlite'
        with sqlite3.connect(self.db) as db:
            db.executescript('CREATE TABLE documents(id INTEGER PRIMARY KEY, filename TEXT);'
                             'CREATE TABLE chunks(id INTEGER PRIMARY KEY, document_id INTEGER, ordinal INTEGER, page TEXT, body TEXT);')
            db.executemany('INSERT INTO documents VALUES (?,?)',[(64,'NeuroAI.pdf'),(101,'review.pdf'),(44,'bess.pdf')])
            db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',[
                (1,64,0,'1','NeuroAI and Beyond: a perspective on embodied machine intelligence and sensorimotor prediction. This is a perspective paper.'),
                (2,64,3,'2','Abstract The article reviews how embodied agents interact with their environments. It is not an interventional clinical trial.'),
                (3,64,6,'3','Introduction We discuss the difference between embodied cognition and learning systems, and possible future directions.'),
                (4,101,2,'1','Abstract This systematic review synthesizes published research on mechanistic targets and clinical questions in autoimmune diseases.'),
                (5,101,7,'2','Introduction The review examines immune pathways and therapies described in current literature. No trial effects are estimated here.'),
                (6,44,12,'2','Methods Study design and participants. The BESS trial was a multicentre blinded randomized sham controlled trial.'),
                (7,44,20,'3','The primary endpoint was the duration of invasive mechanical ventilation in hours from randomisation to extubation.'),
                (8,44,51,'5','Results From enrolment, 232 infants were randomized to two treatment groups.'),
                (9,44,61,'6','The geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86) for the primary result.'),
            ])

    def tearDown(self):
        self.temp.cleanup()

    def test_compare_design_not_effect(self):
        self.assertEqual(requested_slots('比较Focus Pro试验与BESS在研究对象和对照处理上的区别'),('design','overview'))

    def test_general_screen_activity_is_overview(self):
        self.assertEqual(requested_slots('研究中比较了哪些屏幕活动类型？'),('overview',))

    def test_specific_results_stays_results(self):
        self.assertIn('results',requested_slots('比较两个临床试验的实际结果和效应量'))

    def test_abstain_question_does_not_force_results(self):
        self.assertEqual(requested_slots('这篇综述是否报告了随机临床试验的疗效估计？'),('overview',))

    def test_review_without_rct_headings_has_overview_spans(self):
        spans,slots=SQLiteEvidenceHarvester(self.db).harvest([101],'阅读这篇综述的摘要与主要发现')
        self.assertTrue(any(s.section=='overview' for s in spans))
        self.assertTrue(any(s.slot=='overview' and s.status=='candidate' for s in slots))

    def test_nontrial_science_document_no_false_results(self):
        spans,slots=SQLiteEvidenceHarvester(self.db).harvest([64],'阅读NeuroAI文章摘要与研究问题')
        self.assertTrue(spans)
        self.assertFalse(any(s.section=='results' for s in spans))
        self.assertTrue(any(s.slot=='overview' and s.status=='candidate' for s in slots))

    def test_quote_guard_accepts_real_overview_claim(self):
        s=EvidenceSpan('E64_2',64,2,3,'2','overview',
            'Abstract The article reviews how embodied agents interact with their environments. It is not an interventional clinical trial.')
        c={'study_id':64,'slot':'overview','evidence_id':s.evidence_id,
           'quote':s.text,'text':'这篇文章讨论具身智能体与环境之间的交互。'}
        out=ClaimSupportVerifier().review([c],[s]);self.assertEqual(len(out['accepted']),1)
        self.assertFalse(out['medical_claims_verified'])

    def test_overview_cannot_launder_results_claim(self):
        s=EvidenceSpan('E64_2',64,2,3,'2','overview','Abstract The article reviews the field with no intervention result.')
        c={'study_id':64,'slot':'results','evidence_id':s.evidence_id,'quote':s.text,'text':'疗效显著。'}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],'non_results_section')

    def test_significance_wrong_pvalue_rejected(self):
        s=EvidenceSpan('E44_9',44,9,61,'6','results',
            'The adjusted analysis gave a statistical p=0.09 for the association between exposure and outcome.')
        c={'study_id':44,'slot':'results','evidence_id':s.evidence_id,
           'quote':s.text,'text':'调整后的p=0.09说明有显著关联。'}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],
                         'positive_significance_conflicts_with_cited_pvalue')

    def test_numeric_invention_remains_blocked(self):
        s=EvidenceSpan('E44_9',44,9,61,'6','results',
            'The geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
        c={'study_id':44,'slot':'results','evidence_id':s.evidence_id,'quote':s.text,
           'text':'几何均值比为1.99。'}
        self.assertEqual(ClaimSupportVerifier().review([c],[s])['rejected'][0]['check'],
                         'unsupported_numeric_in_claim')

    def test_text_first_supports_overview_task(self):
        spans,slots=SQLiteEvidenceHarvester(self.db).harvest([64],'阅读NeuroAI文章摘要与研究问题')
        llm=QwenJsonClient(completion=lambda s,u:'本文是一篇讨论具身智能和机器学习的观点文章。')
        draft=ClaimDraftGenerator(llm,max_calls=2,strategy='text-first').generate('阅读NeuroAI文章摘要与研究问题',spans,slots)
        self.assertGreaterEqual(len(draft['claims']),1)
        self.assertEqual(draft['claims'][0]['slot'],'overview')
        self.assertEqual(len(ClaimSupportVerifier().review(draft['claims'],spans)['accepted']),1)

    def test_no_unknown_document_invention(self):
        spans,slots=SQLiteEvidenceHarvester(self.db).harvest([101],'阅读综述的研究问题')
        self.assertTrue(all(x.document_id==101 for x in spans))
        self.assertTrue(all(x.document_id==101 for x in slots))

if __name__=='__main__':unittest.main()
