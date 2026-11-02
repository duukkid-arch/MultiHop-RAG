import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mhrag_v13.evidence_slots import EvidenceSpan
from mhrag_v13.e2_field8 import plan,rank_spans,field_first_generate,field_value_matches


def s(cid,text,section='methods'):
    return EvidenceSpan(f'E3_{cid}',3,cid,cid,'2',section,text)

class FakeLLM:
    def __init__(self,answer):self.answer=answer;self.calls=0
    def request_raw(self,sysmsg,user,max_tokens=250):
        self.calls+=1
        return self.answer

class TestFieldFirst(unittest.TestCase):
    def test_plan_multi_fields(self):
        _,fields=plan('患者的年龄和发病时间要求？')
        self.assertEqual([f.name for f in fields],['age','onset'])
    def test_plan_regimen(self):
        _,fields=plan('首次poractant剂量和后续给药方案？')
        self.assertEqual([f.name for f in fields],['initial_dose','followup_dose','interval'])
    def test_plan_effect(self):
        _,fields=plan('第8周联合组TOVA注意比较评分差异和效应量？')
        self.assertEqual([f.name for f in fields],['timepoint','statistic'])
    def test_plan_absolute(self):
        _,fields=plan('能否证明所有儿童停止药物？')
        self.assertEqual(fields[0].name,'study_drug_context')
    def test_primary_definition_over_analysis(self):
        q='主要终点是什么？'
        spans=[s(1,'We conducted the primary outcome analysis on the complete cases.'),
               s(2,'The primary endpoint was the duration of IMV from randomisation to final successful extubation.')]
        self.assertEqual(rank_spans(q,spans)[0]['span'].chunk_id,2)
    def test_TOVA_vs_other_scale(self):
        q='第8周TOVA联合组比较差异和效应量是多少？'
        spans=[s(1,'SNAP-IV week eight (p=0.231, Cohen’s d=0.379), no difference.','results'),
               s(2,'TOVA comparison scores week eight (p=0.021, Cohen’s d=0.609).','results')]
        self.assertEqual(rank_spans(q,spans)[0]['span'].chunk_id,2)
    def test_dose_correct_passes_with_three_fields(self):
        source=s(1,'Initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals for patients.')
        mock=FakeLLM('首次为200 mg/kg，后续100 mg/kg，每12小时一次。')
        out=field_first_generate(mock,'首次poractant剂量和后续给药方案？',[source],4)
        self.assertEqual(set(out['covered_fields']),{'initial_dose','followup_dose','interval'})
        self.assertEqual(out['accepted_count'],1)
        self.assertEqual(mock.calls,1)
    def test_dose_hallucination_not_accepted(self):
        source=s(1,'Initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.')
        out=field_first_generate(FakeLLM('首次200 mg/kg，后续110 mg/kg，每12小时一次。'),
            '首次poractant剂量和后续给药方案？',[source],4)
        self.assertFalse(out['accepted'])
        self.assertEqual(out['diagnostics'][0]['check'],'unsupported_numeric_in_claim')
    def test_no_evidence_no_fabrication(self):
        source=s(1,'Initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.')
        out=field_first_generate(FakeLLM('NO_EVIDENCE'),'首次poractant剂量和后续给药方案？',[source],4)
        self.assertEqual(out['accepted_count'],0)
        self.assertTrue(all(d['status']=='no_evidence' for d in out['diagnostics']))
    def test_references_not_used_when_not_supplied(self):
        source=s(1,'Reference: other study 30 mg/kg, unrelated.')
        out=field_first_generate(FakeLLM('100 mg/kg'),
            '首次poractant剂量和后续给药方案？',[source],4)
        self.assertEqual(out['accepted_count'],0)
    def test_age_wrong_endpoint_rejected_even_if_number_elsewhere(self):
        age=plan('年龄与发病时间要求？')[1][0]
        source='age 18–70 years; 75 percent of participants responded.'
        self.assertFalse(field_value_matches(age,source,'年龄为18–75岁。'))
        self.assertTrue(field_value_matches(age,source,'年龄为18–70岁。'))
    def test_week8_other_endpoint_p_rejected(self):
        stat=plan('第8周TOVA评分差异和效应量是多少？')[1][1]
        source='week eight (p=0.006, Cohen’s d=0.809), overall p=0.005.'
        self.assertTrue(field_value_matches(stat,source,'第8周p=0.006，效应量d=0.809。'))
        self.assertFalse(field_value_matches(stat,source,'第8周p=0.005，效应量d=0.809。'))
    def test_followup_value_bound_to_role(self):
        ff=plan('首次poractant剂量和后续给药方案？')[1][1]
        self.assertFalse(field_value_matches(ff,'Initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.',
                                             '后续给200 mg/kg。'))
        self.assertTrue(field_value_matches(ff,'Initially 200 mg/kg, followed by 100 mg/kg at 12 h intervals.',
                                            '后续给100 mg/kg。'))
    def test_empty_source(self):
        out=field_first_generate(FakeLLM('something'),'主要终点是什么？',[],4)
        self.assertEqual(out['generation_calls'],0)
        self.assertEqual(out['missing_fields'],['outcome_definition'])

if __name__=='__main__':unittest.main()
