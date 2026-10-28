from __future__ import annotations
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_evidence_fallback_v05 import extractive_fallback, _verbatim_window

S=[{'citation':1,'title':'WPS-24-250.pdf','page':1,
'snippet': 'In the nationwide observational cohorts, medium-dose aripiprazole plus high-dose clozapine was associated with decreased hospitalization due to psychosis (aHR=0.68, 95% CI: 0.62-0.75, p<0.0001).'},
{'citation':2,'title':'WPS-24-250.pdf','page':3,'snippet':'The cohort included 22,796 patients.'}]
class TestFallback(unittest.TestCase):
    def test_contains_verbatim_study_result_and_citation(self):
        x=extractive_fallback(S,'无引用')
        self.assertIn('aHR=0.68',x)
        self.assertIn('0.62-0.75',x)
        self.assertIn('[1] WPS-24-250.pdf（PDF 第 1 页）',x)
    def test_does_not_echo_uncited_model_hallucination(self):
        x=extractive_fallback(S,'模型声称氯氮平绝对安全')
        self.assertNotIn('绝对安全',x)
        self.assertIn('不代表模型已经得出可靠的医学结论',x)
    def test_bounded_two_sources(self):
        x=extractive_fallback(S+[{'citation':3,'title':'third','snippet':'third evidence'}])
        self.assertNotIn('third evidence',x)
    def test_no_source(self):
        self.assertIn('没有可用的原文片段',extractive_fallback([]))
    def test_actual_substring(self):
        raw='aaa'*100+' aHR=0.68, 95% CI: 0.62-0.75 '+ 'bbb'*100
        ex=_verbatim_window(raw,maximum=100)
        self.assertIn('aHR=0.68',ex)
        self.assertIn('0.62-0.75',ex)
    def test_no_model_claims_added(self):
        x=extractive_fallback([{'citation':1,'title':'test.pdf','page':2,'snippet':'The evidence is incomplete.'}])
        self.assertIn('The evidence is incomplete.',x)
        self.assertNotIn('有统计学显著性',x)
if __name__=='__main__':unittest.main(verbosity=2)
