import unittest
from mhrag_v13.question_coverage import evaluate, CoverageController, question_spec


def e(text,doc=44,slot='overview',eid=None):
    return {'claim':{'text':text,'study_id':doc,'slot':slot,'evidence_id':eid or ('E'+str(doc)+'_1')}}


class QuestionCoverageV09Tests(unittest.TestCase):
    def test_universal_background_not_answer(self):
        r,a=evaluate('BESS是否证明给每位重症患儿使用药物必然有益？', [e('Bronchiolitis是一种婴儿常见的病毒性疾病')],{'complete':True})
        self.assertEqual(r['status'],'no_task_aligned_claim')
    def test_universal_direct_negative_is_candidate_only(self):
        r,a=evaluate('BESS是否证明给每位患儿用药必然有益？',[e('这项试验没有证明每位患儿必然获益。')],{'complete':True})
        self.assertEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
        self.assertEqual(r['medical_accuracy'],'NOT_EVALUATED')
    def test_comparison_one_study_is_incomplete(self):
        r,_=evaluate('比较两项试验在研究对象和对照措施上的区别', [e('这项研究招募婴儿并设置安慰剂对照。',44,'design')],{'complete':True})
        self.assertIn('missing_two_study_specific_evidence',r['reasons'])
    def test_two_studies_still_need_comparison(self):
        r,_=evaluate('比较两项试验研究对象和对照措施的区别', [e('研究对象是婴儿。',44,'design'),e('研究对象是儿童。',55,'design')],{'complete':True})
        self.assertIn('no_explicit_comparison_conclusion',r['reasons'])
    def test_two_studies_explicit_comparison(self):
        r,_=evaluate('比较两项试验研究对象和对照措施的区别',[e('第一篇研究纳入婴儿和安慰剂对照。',44,'design'),e('另一篇研究纳入儿童，对照组使用哌甲酯，两者人群和对照方式不同。',55,'design')],{'complete':True})
        self.assertEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
    def test_dedup(self):
        r,a=evaluate('比较两项试验的研究对象和对照', [e('研究对象是婴儿且有安慰剂对照。'),e('研究对象是婴儿且有安慰剂对照。'),e('受试者为儿童，对照方式不同。',55)],{'complete':True})
        self.assertEqual(r['duplicate_count'],1)
        self.assertEqual(len(a),2)
    def test_catalog_bare_review_insufficient(self):
        r,_=evaluate('找到名为 abc.pdf 的论文，报告疾病研究设计主要发现',[e('这是一篇综述文章',101)],{'complete':True})
        self.assertIn('missing_main_finding',r['reasons'])
    def test_catalog_finding_incomplete(self):
        r,_=evaluate('定位文件abc.pdf，说明研究问题类型和主要发现',[e('论文研究牛皮癣的治疗机制。',101),e('这是一篇综述，结论指出还需高质量证据。',101)],{'complete':True})
        self.assertEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
    def test_enumeration_background_rejected(self):
        r,_=evaluate('研究中比较了哪些屏幕活动类型？',[e('研究屏幕使用与心理社会发展关联。',59)],{'complete':True})
        self.assertEqual(r['status'],'no_task_aligned_claim')
    def test_enumeration_lists_sources(self):
        r,_=evaluate('研究中比较了哪些屏幕活动类型？',[e('研究比较了电视、电子游戏和社交媒体三类屏幕活动。',59)],{'complete':True})
        self.assertEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
    def test_provisional_never_complete(self):
        r,_=evaluate('这项研究有何结果？',[e('研究结果显示改善。',59,'results')], {'complete':False},{'provisional':True})
        self.assertEqual(r['status'],'partial_task_coverage')
    def test_no_medical_truth_from_quote(self):
        r,_=evaluate('BESS是否证明每位患儿必然获益？',[e('这项研究不能证明每位患儿必然获益。')],{'complete':True})
        self.assertNotIn('verified',r['status'].casefold().replace('not_medically_verified',''))
    def test_generic_not_substitute_for_evidence_limit(self):
        r,_=evaluate('是否可以计算全部治疗的统一死亡率降低？',[e('讨论欧洲指南支持TAVR。',1)],{'complete':False},{'provisional':True})
        self.assertEqual(r['status'],'no_task_aligned_claim')
    def test_coverage_controller_downgrade_no_fabrication(self):
        class Base:
            def run(self,q,*,evidence_only=False):
                return {'answer':'- Bronchiolitis是一种疾病。','decision':{'decision':'provisional_answer'},
                    'claims':{'accepted':[e('Bronchiolitis是一种婴儿疾病。')]},
                    'identity':{'complete':True},'routing':{'provisional':False}}
        r=CoverageController(Base()).run('BESS是否证明所有患儿必然获益？')
        self.assertEqual(r['decision']['decision'],'abstain')
        self.assertIn('未找到足以直接回应',r['answer'])
        self.assertEqual(r['upstream_decision']['decision'],'provisional_answer')
    def test_evidence_only_keeps_payload(self):
        class Base:
            def run(self,q,*,evidence_only=False):return {'answer':'原始证据审计','claims':{'accepted':[]}}
        r=CoverageController(Base()).run('任何问题',evidence_only=True)
        self.assertEqual(r['answer'],'原始证据审计')
    def test_unicode_case_and_chinese_punctuation(self):
        r,_=evaluate('能否计算统一死亡率降低？',[e('不能计算统一死亡率降低百分比，缺少可比的结局定义。',1)],{'complete':True})
        self.assertEqual(r['status'],'candidate_task_covered_NOT_MEDICALLY_VERIFIED')
    def test_explicit_compare_without_sources_never_complete(self):
        r,_=evaluate('比较文章A和B的疾病范围',[],{'complete':True})
        self.assertEqual(r['status'],'no_task_aligned_claim')

if __name__ == '__main__':unittest.main()
