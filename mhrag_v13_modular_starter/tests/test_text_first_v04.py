"""Live-failure-derived regressions: natural output, no JSON, no cite IDs."""
import json
import unittest
from mhrag_v13.evidence_slots import EvidenceSpan,Slot
from mhrag_v13.grounded_answers import (
    QwenJsonClient,ClaimDraftGenerator,ClaimSupportVerifier,
    parse_single_sentence_reply,
)

BESS_TEXT=('Results The geometric mean ratio of the poractant alfa group compared '
           'with sham was 1.02 (95% CI 0.84 to 1.24; p=0.86), indicating that '
           'the treatment effect on duration of mechanical ventilation was not significant.')
FOCUS_TEXT=('Results On TOVA attention comparison scores at week eight, '
            'the combination group showed greater improvement (p=0.006, Cohen d=0.809) '
            'than the control group in this ADHD randomized trial.')

def samples():
    a=EvidenceSpan('E44_4882',44,4882,62,'5','results',BESS_TEXT)
    b=EvidenceSpan('E55_6019',55,6019,4,'1','abstract_results',FOCUS_TEXT)
    slots=[Slot(44,'results','candidate',(a.evidence_id,),'candidate'),
           Slot(55,'results','candidate',(b.evidence_id,),'candidate')]
    return [a,b],slots

class TextFirstV04(unittest.TestCase):
    def test_natural_language_two_papers_no_json(self):
        spans,slots=samples()
        calls=[]
        def completion(system,user):
            d=json.loads(user);calls.append((system,d))
            self.assertIn('original_evidence_quote',d)
            self.assertNotIn('quote_options',d)
            if d['study_id']==44:
                return '研究报告几何均值比为1.02，p=0.86。'
            return 'ADHD 联合干预组TOVA注意力比较分数改善更显著（p=0.006）。'
        gen=ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=2,strategy='text-first')
        result=gen.generate('比较两项研究结果',spans,slots)
        self.assertEqual(result['generation_calls'],2)
        self.assertEqual(len(result['claims']),2)
        self.assertTrue(all(d['status']=='drafted' for d in result['diagnostics']))
        self.assertTrue(all(d['stage']=='slot_text_first' for d in result['diagnostics']))
        self.assertEqual(len(ClaimSupportVerifier().review(result['claims'],spans)['accepted']),2)
        self.assertEqual([r['study_id'] for r in result['claims']],[44,55])
        self.assertTrue(all('CLAIM|' not in c['quote'] for c in result['claims']))

    def test_no_evidence_no_invented_claim(self):
        spans,slots=samples()
        gen=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'NO_EVIDENCE'),max_calls=2,strategy='text-first')
        r=gen.generate('结果',spans,slots)
        self.assertEqual(r['claims'],[])
        self.assertTrue(all(d['status']=='empty_claims' for d in r['diagnostics']))

    def test_refusal_no_invented_claim(self):
        spans,slots=samples()
        gen=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'无法根据提供的证据确定主要疗效。'),max_calls=1,strategy='text-first')
        r=gen.generate('结果',spans,slots)
        self.assertEqual(r['claims'],[])

    def test_blank_reply_diagnostic_instead_of_decodeerror(self):
        spans,slots=samples()
        gen=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'  '),max_calls=1,strategy='text-first')
        r=gen.generate('结果',spans,slots)
        self.assertEqual(r['diagnostics'][0]['error_message'],'empty_model_reply')
        self.assertEqual(r['diagnostics'][0]['response_chars'],2)
        self.assertEqual(r['diagnostics'][0]['finish_reason'],'mock_completion')

    def test_wrong_numeric_is_rejected(self):
        spans,slots=samples()
        gen=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'这项研究的几何均值比是9.99。'),max_calls=1,strategy='text-first')
        r=gen.generate('结果',spans,slots)
        self.assertEqual(r['diagnostics'][0]['status'],'all_failed_precheck')
        self.assertEqual(len(ClaimSupportVerifier().review(r['claims'],spans)['rejected']),1)

    def test_bound_quote_is_source_substring(self):
        spans,slots=samples()
        r=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'研究报告几何均值比为1.02，p=0.86。'),
                              max_calls=1,strategy='text-first').generate('结果',spans,slots[:1])
        claim=r['claims'][0]
        self.assertEqual(claim['evidence_id'],'E44_4882')
        self.assertIn(claim['quote'],BESS_TEXT)

    def test_no_cross_study_identity_control(self):
        spans,slots=samples()
        def completion(_,user):
            d=json.loads(user)
            self.assertEqual(d['study_id'],44)
            return '研究报道几何均值比为1.02。'
        gen=ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=1,strategy='text-first')
        r=gen.generate('结果',spans,slots[:1])
        self.assertEqual(r['claims'][0]['study_id'],44)
        self.assertEqual(r['claims'][0]['evidence_id'],'E44_4882')

    def test_multiline_text_refused(self):
        quote={'evidence_id':'E44_4882','quote':BESS_TEXT}
        with self.assertRaisesRegex(ValueError,'not_one_short_sentence'):
            parse_single_sentence_reply('第一条事实。\n第二条事实。',44,'results',quote)

    def test_json_response_refused_not_decoded(self):
        quote={'evidence_id':'E44_4882','quote':BESS_TEXT}
        with self.assertRaisesRegex(ValueError,'model_replied_json'):
            parse_single_sentence_reply('{"claims":[]}',44,'results',quote)

    def test_quotes_always_program_bound(self):
        quote={'evidence_id':'E44_4882','quote':BESS_TEXT}
        c=parse_single_sentence_reply('答案：研究报告两组结果差异。',44,'results',quote)['claims'][0]
        self.assertEqual(c['quote'],BESS_TEXT)
        self.assertEqual(c['study_id'],44)

    def test_extra_sentence_rejected(self):
        quote={'evidence_id':'E44_4882','quote':BESS_TEXT}
        with self.assertRaisesRegex(ValueError,'multiple_sentences'):
            parse_single_sentence_reply('结果正常。结论进一步证明疗效。',44,'results',quote)

    def test_invalid_strategy_rejected(self):
        with self.assertRaises(ValueError):
            ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'OK'),strategy='free')

if __name__=='__main__':unittest.main()
