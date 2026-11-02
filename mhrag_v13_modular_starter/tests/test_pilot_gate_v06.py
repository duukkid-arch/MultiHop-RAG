from __future__ import annotations
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import v13_pilot_runner as r

class TestPilotGateV06(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).resolve().parents[1] / 'benchmarks' / 'pilot16_dev_questions.json'
        self.cases = r.load_cases(self.fixture)

    def test_cases_are_sanitized_and_16(self):
        self.assertEqual(len(self.cases), 16)
        self.assertTrue(all(set(x)=={'case_id','question','development_phase'} for x in self.cases))
        self.assertFalse(any('gold' in str(x).lower() for x in self.cases))

    def test_plan_paired_and_deterministic(self):
        a=r.make_plan(self.cases,42)
        self.assertEqual(a,r.make_plan(self.cases,42))
        self.assertEqual(len(a),32)
        self.assertEqual(len({(x['case_id'],x['arm']) for x in a}),32)
        self.assertTrue(all({x['arm'] for x in a[i:i+2]}=={'F','G'} for i in range(0,32,2)))

    def test_gold_leak_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source=json.loads(self.fixture.read_text())
            source['cases'][0]['gold_ids']=[44]
            path=Path(d)/'cases.json';path.write_text(json.dumps(source))
            with self.assertRaises(ValueError):r.load_cases(path)

    def test_resume_reject_duplicate_or_order_mismatch(self):
        plan=r.make_plan(self.cases,42)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'results.jsonl'
            path.write_text(json.dumps({**plan[1],'status':'ok'})+'\n')
            with self.assertRaises(ValueError):r.read_existing(path,plan)
            path.write_text(json.dumps({**plan[0],'status':'ok'})+'\n'+json.dumps({**plan[0],'status':'ok'})+'\n')
            with self.assertRaises(ValueError):r.read_existing(path,plan)

    def test_resume_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'results.jsonl';path.write_text('{bad\n')
            with self.assertRaises(ValueError):r.read_existing(path,r.make_plan(self.cases,42))

    def test_resume_normal_reads(self):
        plan=r.make_plan(self.cases,42)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'results.jsonl'
            path.write_text('\n'.join(json.dumps({**x,'status':'ok'}) for x in plan[:2])+'\n')
            self.assertEqual(len(r.read_existing(path,plan)),2)

    def test_standard_baseline_agentic_false(self):
        class Searcher:
            def search(self, q, *, agentic=False):
                assert not agentic
                return type('Result',(),{'answer':'fake answer','sources':[]})()
        out=r.run_job({'case_id':'x','arm':'F'},{'x':'a question'},Searcher(),None,'qwen')
        self.assertEqual(out['status'],'ok')
        self.assertEqual(out['answer'],'fake answer')
        self.assertEqual(out['medical_claims_verified'],False)

    def test_grounded_does_not_imply_medical_gold(self):
        class Grounded:
            def run(self,q,*,evidence_only):
                return {'answer':'safe','claims':{'accepted':[{'claim':{'study_id':44}}],'rejected':[]},
                        'evidence':{'api_doc_ids':[44],'local_doc_ids':[44]},
                        'retrieval':{'http_calls':1}, 'slots':[], 'decision':{'decision':'answer'},
                        'generation_status':'drafted','generation_calls':1,
                        'draft_claim_count':1,'rejected_claim_count':0}
        out=r.run_job({'case_id':'x','arm':'G'},{'x':'a question'},None,Grounded(),'qwen')
        self.assertEqual(out['accepted_study_ids'],[44])
        self.assertFalse(out['medical_claims_verified'])
        self.assertTrue(out['human_medical_review_required'])

    def test_runtime_error_recorded(self):
        class Searcher:
            def search(self,*a,**kw):raise TimeoutError('slow')
        out=r.run_job({'case_id':'x','arm':'F'},{'x':'a question'},Searcher(),None,'qwen')
        self.assertEqual(out['status'],'error')
        self.assertEqual(out['error_type'],'TimeoutError')

    def test_summary_no_false_clinical_success(self):
        plan=r.make_plan(self.cases,42)
        rows=[{**j,'status':'ok','wall_ms':15,'accepted_claim_count':0,'answer':'abstain',
               'decision':{'decision':'partial_or_abstain'}} for j in plan]
        s=r.summary(rows,plan)
        self.assertEqual(s['paired_cases_completed'],16)
        self.assertEqual(s['medical_accuracy'],'NOT_EVALUATED')
        self.assertEqual(s['arms']['G']['cases_with_accepted_claim'],0)

    def test_source_hash_detect_change(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x';f.write_bytes(b'first')
            a=r.sha256_file(f);f.write_bytes(b'second')
            self.assertNotEqual(a,r.sha256_file(f))

if __name__=='__main__':unittest.main()
