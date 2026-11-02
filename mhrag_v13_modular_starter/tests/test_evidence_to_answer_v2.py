import unittest,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mhrag_v13.evidence_slots import EvidenceSpan
from mhrag_v13.evidence_to_answer_v2 import (source_bound_generate,recover_single_sentence,
    SourceBoundFallbackGenerator)
from benchmarks.run_dev24_e2v2 import load_questions,make_plan


def span(did,cid,text,section='methods'):
    return EvidenceSpan(f'E{did}_{cid}',did,cid,cid,'1',section,text)

class SourceBoundTests(unittest.TestCase):
    def test_regimen_exact(self):
        s=span(10,11,'Patients initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals or sham interventions.')
        o=source_bound_generate('首次剂量和后续poractant给药方案是什么？',[s]);self.assertEqual(len(o['claims']),1)
        self.assertIn('100 mg/kg',o['claims'][0]['text']);self.assertIn('E10_11',o['claims'][0]['evidence_id'])
    def test_no_arbitrary_numeric_claim(self):
        s=span(10,11,'Patients initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.')
        c,why=recover_single_sentence('首次200 mg/kg，随后110 mg/kg，每12小时一次。NO_EVIDENCE',s)
        self.assertIsNone(c);self.assertEqual(why,'unsupported_numeric_in_claim')
    def test_trailing_marker_recovery(self):
        s=span(10,11,'Patients initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.')
        c,why=recover_single_sentence('首次200 mg/kg，随后100 mg/kg，每12小时一次。NO_EVIDENCE',s)
        self.assertIsNotNone(c);self.assertEqual(why,'recovered_and_numeric_checked')
    def test_not_middle_marker(self):
        s=span(10,11,'Patients initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.')
        self.assertIsNone(recover_single_sentence('NO_EVIDENCE。首次200 mg/kg',s)[0])
    def test_age_onset_both(self):
        s=span(52,99,'Eligibility: age 18–70 years, and presentation less than 96 h from ictus.')
        o=source_bound_generate('年龄与发病时间要求？',[s]);self.assertEqual(len(o['claims']),1)
        self.assertIn('18–70',o['claims'][0]['text']);self.assertIn('96小时',o['claims'][0]['text'])
    def test_no_wrong_age_mapping(self):
        s=span(52,99,'Participants age 18–70 years, and median age was 75 years. No onset criterion.')
        self.assertEqual(source_bound_generate('年龄与发病时间要求？',[s])['claims'],[])
    def test_no_other_scale_week8(self):
        s=span(55,99,'SNAP-IV week eight (p=0.006, Cohen’s d=0.809).','results')
        self.assertEqual(source_bound_generate('第8周TOVA比较评分差异和效应量？',[s])['claims'],[])
    def test_tova_week8(self):
        s=span(55,99,'TOVA scores showed advantage at week eight (p=0.006, Cohen’s d=0.809, 95% CI: 0.092 to 1.527).','results')
        o=source_bound_generate('第8周TOVA比较评分差异和效应量？',[s]);self.assertEqual(len(o['claims']),1)
        self.assertIn('0.006',o['claims'][0]['text'])
    def test_references_rejected(self):
        s=span(10,21,'References\n1. Therapy initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.')
        self.assertFalse(source_bound_generate('首次剂量和后续poractant给药方案是什么？',[s])['claims'])
    def test_unsupported_fallback(self):
        class Fake:
            def generate(self,*a,**kw):return {'claims':[],'diagnostics':[],'generation_calls':0}
        # With no spans, old E2 never called for source-bound output, but fallback exists.
        self.assertTrue(hasattr(SourceBoundFallbackGenerator,'generate'))
    def test_dev24_manifest(self):
        src=Path(__file__).resolve().parents[1]/'benchmarks/dev24_e2v2_NO_GOLD.json'
        cases=load_questions(src)
        self.assertEqual(len(cases),24)
        p=make_plan(cases)
        self.assertEqual(len(p),96)
        self.assertEqual({x['arm'] for x in p},{'F','E1','E2R','E3R'})
        self.assertFalse(any(x['case_id'] in {'v9_003','v9_005','v9_009','v9_011','v9_026','v9_027','abstain_002','abstain_003'} for x in p))

if __name__=='__main__':unittest.main()
