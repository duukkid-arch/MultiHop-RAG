import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_guard_v04 import strict_problem, validate_and_repair

SOURCES = [{'snippet':'Adjusted hazard ratio (aHR=0.68, 95% CI 0.62-0.75); associated with lower hospitalizations.', 'citation':1}]
def base(text, n):
    import re
    ids=[int(i) for i in re.findall(r'\[(\d+)\]', text)]
    return '无效引用编号' if any(i<1 or i>n for i in ids) else None
def evidence(q, a, s): return None

class TestV04(unittest.TestCase):
    def test_short_uncited_fails(self):
        self.assertIsNotNone(strict_problem('aHR = 0.68, 95% CI: 0.62-0.75',SOURCES))
    def test_cited_correct_effect_passes(self):
        self.assertIsNone(strict_problem('观察到较低住院风险（aHR=0.68）[1]',SOURCES))
    def test_cited_fabricated_effect_fails(self):
        self.assertIn('无法核实',strict_problem('aHR=0.12[1]',SOURCES))
    def test_short_abstention_permitted(self):
        self.assertIsNone(strict_problem('所给片段证据不足，无法确定。',SOURCES))
    def test_retry_once(self):
        outputs=[]
        def answer(q,src,history=None):
            outputs.append(q)
            return '观察到较低住院风险（aHR=0.68）[1]'
        text,issue=validate_and_repair('是否有效',SOURCES,'aHR=0.68',answer,base,evidence)
        self.assertIsNone(issue)
        self.assertIn('[1]',text)
        self.assertEqual(len(outputs),1)
    def test_retry_failure_stays_blocked(self):
        def answer(q,src,history=None): return 'aHR=0.68'
        _,issue=validate_and_repair('是否有效',SOURCES,'aHR=0.68',answer,base,evidence)
        self.assertIsNotNone(issue)
    def test_no_retry_for_unsupported_effect(self):
        def answer(*args,**kwargs): self.fail('should not retry')
        _,issue=validate_and_repair('是否有效',SOURCES,'aHR=0.12[1]',answer,base,evidence)
        self.assertIsNotNone(issue)
    def test_citation_range(self):
        _,issue=validate_and_repair('是否有效',SOURCES,'观察到降低[aHR=0.68][9]',lambda *a,**k:'aHR=0.68 [1]',base,evidence)
        self.assertIsNone(issue)

if __name__ == '__main__': unittest.main(verbosity=2)
