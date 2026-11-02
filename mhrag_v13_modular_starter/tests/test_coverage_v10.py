import json
import unittest
from pathlib import Path
from mhrag_v13.question_coverage_v10 import evaluate, question_kind, _is_purpose_only


def claim(doc,txt,slot='overview'):
    return {'claim': {'study_id': doc,'slot':slot,'text':txt,'evidence_id':f'E{doc}_100','quote':'Some original quote 123456789'}}

class CoverageV10Tests(unittest.TestCase):
    def test_aim_only_not_main_finding(self):
        q='请定位这篇文章的研究问题、研究类型和主要发现'
        report,_=evaluate(q,[claim(109,'这篇综述旨在总结自然观察研究中双相障碍抑郁症状的药物治疗结果')],{'complete':True},{'provisional':False})
        self.assertNotEqual(report['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
        self.assertIn('missing_asserted_main_finding_not_just_study_aim',report['reasons'])
    def test_finding_requires_actual_assertion(self):
        self.assertTrue(_is_purpose_only('这篇综述旨在总结临床治疗的结果'))
        self.assertFalse(_is_purpose_only('研究结果显示了显著差异'))
    def test_universal_overclaim_generic_rct_not_sufficient(self):
        r,_=evaluate('是否证明所有儿童停止药物？',[claim(55,'这项研究为单中心双盲随机对照试验','design')],{'complete':True},{})
        self.assertNotEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
    def test_evidence_limit_has_precedence_over_everyone(self):
        q='是否可以计算再生医学相对于全部传统再灌注方法的统一死亡率降低百分比？'
        self.assertEqual(question_kind(q),'evidence_limit')
        r,_=evaluate(q,[claim(1,'欧洲指南支持更低风险的患者')],{'complete':True},{})
        self.assertIn('no_direct_evidence_limit_conclusion',r['reasons'])
    def test_comparison_study_count(self):
        q='比较Focus Pro试验和BESS试验在受试者及对照方面的区别。'
        r,_=evaluate(q,[claim(44,'婴儿毛细支气管炎病例接受对照处理','design'),claim(55,'ADHD随机试验的对照组接受MPH','design')],{'complete':True},{})
        self.assertIn('missing_explicit_two_study_contrast',r['reasons'])
    def test_purpose_alone_not_study_specific(self):
        q='比较甲综述与乙综述有何不同？'
        r,_=evaluate(q,[claim(44,'这篇综述旨在总结研究结果'),claim(55,'本综述旨在总结研究发现')],{'complete':True},{})
        self.assertIn('missing_two_study_specific_content',r['reasons'])
    def test_partial_identity_remains_partial(self):
        r,_=evaluate('比较两篇文献的研究人群和主要终点', [claim(44,'研究采用随机对照设计')], {'complete':False},{'provisional':False})
        self.assertNotEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
    def test_no_medical_claim(self):
        r,_=evaluate('这篇综述的研究问题、类型及主要发现是什么？',[],{'complete':True},{})
        self.assertEqual(r['medical_accuracy'],'NOT_EVALUATED')
    def test_severe_conflation_flags(self):
        q='急性B型主动脉夹层与2025年ACS急性冠脉综合征指南是否相同？'
        r,_=evaluate(q,[claim(83,'主动脉夹层与ACS急性冠脉综合征是同一种疾病')],{'complete':True},{})
        self.assertIn('potential_disease_identity_conflation',r['reasons'])
    def test_dataset_sanitized_91(self):
        p=Path(__file__).resolve().parents[1]/'benchmarks'/'dev91_questions_NO_GOLD.json'
        d=json.loads(p.read_text())
        self.assertEqual(len(d['cases']),91)
        self.assertEqual({*map(lambda c:tuple(sorted(c)),d['cases'])},{('case_id','development_phase','question')})
        text=p.read_text()
        self.assertNotIn('gold_ids',text)
        self.assertNotIn('reference',text)

if __name__=='__main__':unittest.main()
