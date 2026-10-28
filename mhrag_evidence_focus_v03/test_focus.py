"""Unit tests independent of GPU and real LLM. Execute before and after installing."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mhrag_evidence_focus_v03.answer_guard_v03 import evidence_problem

class GuardTests(unittest.TestCase):
    def test_block_false_no_effect_claim(self):
        sources=[{'snippet':'In both countries aHR=0.68, 95% CI: 0.62-0.75 was associated with reduced hospitalizations.'}]
        q='精神病复发住院风险是否降低？'
        self.assertIsNotNone(evidence_problem(q,'文献未详细描述复发住院风险的具体关系。[1]',sources))
    def test_allow_correct_claim(self):
        src=[{'snippet':'aHR=0.68, 95% CI:0.62-0.75'}]
        self.assertIsNone(evidence_problem('住院风险','观察到较低风险 (aHR=0.68) [1]',src))
    def test_no_block_non_numeric_question(self):
        src=[{'snippet':'clozapine was studied but no statistical effect was given'}]
        self.assertIsNone(evidence_problem('氯氮平是什么','文献未详细描述其住院风险。[1]',src))
    def test_numeric_priority(self):
        import importlib.util
        pkg=Path(__file__).parent.parent/'mhrag_gpu_chat_musa'/'llm.py'
        # Run independently against a copy of this package if v0.3 isn't installed.
        if not pkg.exists() or 'def _source_priority' not in pkg.read_text():self.skipTest('v0.3 not installed')
        from mhrag_gpu_chat_musa import llm
        intro={'title':'study.pdf','page':1,'snippet':'Thus far only one such study suggested lower rates.'}
        result={'title':'study.pdf','page':3,'snippet':'Main outcome: adjusted hazard ratio (aHR=0.68, 95% CI 0.62-0.75) in 22796 people.'}
        with patch.object(llm,'call',return_value='风险较低(aHR=0.68)[2]') as mock:
            llm.answer('住院风险和HR', [intro,result]); message=mock.call_args.args[0][0]['content']
        self.assertLess(message.index('[2] 文件:'),message.index('[1] 文件:'))
        self.assertIn('0.62-0.75',message)

if __name__=='__main__':unittest.main(verbosity=2)
