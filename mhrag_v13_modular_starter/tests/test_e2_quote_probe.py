import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mhrag_v13.e2_quote_probe import parse_exact_quote,QuoteProbe,verify_quote,_candidate_tasks
from mhrag_v13.evidence_slots import EvidenceSpan

SRC=('Methods. In the trial, patients received a dose of 30 mg per day '
     'for eight weeks. Clinicians monitored adherence with daily calls. ')
QUOTE='In the trial, patients received a dose of 30 mg per day for eight weeks.'
S=EvidenceSpan('E44_9001',44,9001,20,'5','methods',SRC)

class MockLLM:
    def __init__(self,answer):self.answer=answer;self.calls=0
    def request_raw(self,system,user,max_tokens=350):
        self.calls+=1
        return self.answer

class TestQuoteProbe(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(parse_exact_quote(QUOTE,SRC),(QUOTE,'verbatim'))
    def test_quote_label(self):
        self.assertEqual(parse_exact_quote('QUOTE: '+QUOTE,SRC)[1],'verbatim')
    def test_bad_copy(self):
        self.assertEqual(parse_exact_quote(QUOTE.replace('30','35'),SRC)[1],'not_verbatim')
    def test_no_evidence(self):
        self.assertEqual(parse_exact_quote('NO_EVIDENCE',SRC),(None,'no_evidence'))
    def test_multiple_lines_rejected(self):
        self.assertEqual(parse_exact_quote(QUOTE+'\nExplanation',SRC)[1],'invalid_format')
    def test_empty(self):
        self.assertEqual(parse_exact_quote('',SRC)[1],'empty_response')
    def test_no_json(self):
        self.assertEqual(parse_exact_quote('{"quote":"foo"}',SRC)[1],'invalid_format')
    def test_needs_subtarget(self):
        self.assertTrue(_candidate_tasks('给药剂量是多少？',[S]))
    def test_accepted_verbatim(self):
        m=MockLLM(QUOTE);r=QuoteProbe(m,max_calls=2).run('给药剂量是多少？',[S])
        self.assertEqual(r['accepted_quote_count'],1)
        self.assertEqual(r['generation_calls'],1)
    def test_fake_quote_rejected(self):
        m=MockLLM(QUOTE.replace('30','35'))
        r=QuoteProbe(m,max_calls=2).run('给药剂量是多少？',[S])
        self.assertEqual(r['accepted_quote_count'],0)
        self.assertEqual(r['diagnostics'][0]['status'],'not_verbatim')
    def test_noevidence_not_invented(self):
        r=QuoteProbe(MockLLM('NO_EVIDENCE'),max_calls=2).run('给药剂量是多少？',[S])
        self.assertEqual(r['accepted_quote_count'],0)
        self.assertEqual(r['generation_calls'],1)
    def test_wrong_target_not_accepted(self):
        # Dates should not pass with a dose-only number.
        bad=EvidenceSpan('E44_9002',44,9002,3,'1','overview',SRC)
        # Force function with actual target to verify no unsupported claims.
        from mhrag_v13.e2_repair_trial import repair_targets
        dt=next(t for t in repair_targets('何时举办研讨会？') if t.name=='date')
        ok,reason=verify_quote(QUOTE,bad,dt)
        self.assertFalse(ok)
        self.assertEqual(reason,'not_task_focused')
    def test_invalid_budget(self):
        with self.assertRaises(ValueError):QuoteProbe(MockLLM(QUOTE),0)

if __name__=='__main__':unittest.main()
