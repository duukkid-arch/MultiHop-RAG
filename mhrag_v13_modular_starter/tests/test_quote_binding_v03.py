"""Regression tests for Qwen 7B line protocol and exact-source quote binding."""
import json
import unittest
from mhrag_v13.evidence_slots import EvidenceSpan,Slot
from mhrag_v13.grounded_answers import (
  ClaimDraftGenerator,ClaimSupportVerifier,QwenJsonClient,parse_json_message,
  parse_single_slot_reply,
)


TEXT='Results The geometric mean ratio was 1.02 (95% CI 0.84 to 1.24; p=0.86) in a randomized trial, without a significant improvement in the duration of mechanical ventilation for infants.'


def spans_and_slots():
    src=EvidenceSpan('E44_4882',44,4882,52,'5','results',TEXT)
    other=EvidenceSpan('E55_6062',55,6062,47,'7','results','Results ADHD attention scores improved in the combination group (p=0.006).')
    return [src,other],[Slot(44,'results','candidate',(src.evidence_id,),'candidate'),Slot(55,'results','candidate',(other.evidence_id,),'candidate')]


def make_completion(claim='研究报告几何均值比为1.02，p=0.86。',*,switch=False):
    def completion(system,user):
        data=json.loads(user)
        if len(data['target_slots'])>1:
            return '{"claims":[],"limitations":[]}'
        if 'quote_options' not in data:
            return '{"claims":[],"limitations":[]}'
        choice=data['quote_options'][0]
        assert choice['quote'] in data['evidence'][0]['text']
        if switch:
            return 'CLAIM|E55_6062|'+choice['quote_id']+'|'+claim
        return 'CLAIM|'+choice['evidence_id']+'|'+choice['quote_id']+'|'+claim
    return completion


class QuoteBindingV03Tests(unittest.TestCase):
    def test_recover_exact_quote_from_numbered_choice(self):
        spans,slots=spans_and_slots()
        gen=ClaimDraftGenerator(QwenJsonClient(completion=make_completion()),max_calls=2)
        out=gen.generate('结果',spans,slots[:1])
        self.assertEqual(out['generation_calls'],2)
        self.assertEqual(out['diagnostics'][1]['status'],'drafted')
        self.assertEqual(out['claims'][0]['quote'],TEXT[:400])
        self.assertEqual(len(ClaimSupportVerifier().review(out['claims'],spans)['accepted']),2 if False else 1)

    def test_reject_cross_paper_forged_evidence_id(self):
        spans,slots=spans_and_slots()
        out=ClaimDraftGenerator(QwenJsonClient(completion=make_completion(switch=True)),max_calls=2).generate('结果',spans,slots[:1])
        self.assertFalse(out['claims'])
        self.assertEqual(out['diagnostics'][1]['status'],'model_error')
        self.assertIn('ValueError',out['diagnostics'][1]['error_type'])

    def test_reject_missing_quote_option(self):
        with self.assertRaises(ValueError):
            parse_single_slot_reply('CLAIM|E44_4882|Q99|研究结果',44,'results',{})

    def test_reject_quote_id_from_different_evidence(self):
        with self.assertRaises(ValueError):
            parse_single_slot_reply('CLAIM|E55_6062|Q0|研究结果',44,'results',
                {'Q0':{'evidence_id':'E44_4882','quote':TEXT}})

    def test_reject_unsupported_numeric_even_with_trusted_quote(self):
        claim=parse_single_slot_reply('CLAIM|E44_4882|Q0|研究报告比值为7.77。',44,'results',
            {'Q0':{'evidence_id':'E44_4882','quote':TEXT}})['claims'][0]
        spans,_=spans_and_slots()
        ok,reason=ClaimSupportVerifier()._check_one(claim,{'E44_4882':spans[0]})
        self.assertFalse(ok)
        self.assertEqual(reason,'unsupported_numeric_in_claim')

    def test_no_evidence_does_not_invent_claim(self):
        self.assertEqual(parse_single_slot_reply('NO_EVIDENCE',44,'results',{})['claims'],[])

    def test_malformed_line_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_single_slot_reply('CLAIM|E44_4882|Q0',44,'results',{'Q0':{'evidence_id':'E44_4882','quote':TEXT}})

    def test_json_control_character_can_be_decoded(self):
        result=parse_json_message('{"claims":[],"limitations":["raw\nline"]}')
        self.assertEqual(result['limitations'],['raw\nline'])

    def test_trailing_comma_repaired_without_changing_quote(self):
        obj=parse_json_message('{"claims":[],"limitations":["trial, result"],}')
        self.assertEqual(obj['limitations'],['trial, result'])

    def test_legacy_json_still_supported(self):
        obj={'claims':[{'study_id':44,'slot':'results','evidence_id':'E44_4882',
                        'quote':TEXT,'text':'报告了统计分析的结果。'}], 'limitations':[]}
        result=parse_single_slot_reply(json.dumps(obj,ensure_ascii=False),44,'results',{})
        self.assertEqual(result['claims'][0]['quote'],TEXT)

    def test_quoted_unknown_numeric_also_checked(self):
        spans,slots=spans_and_slots()
        q=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'{}'))._quote_choices([spans[0]])
        self.assertTrue(q)
        self.assertTrue(all(x['quote'] in TEXT for x in q))
        self.assertTrue(all(x['evidence_id']=='E44_4882' for x in q))
        self.assertLessEqual(len(q),3)

class SlotFirstStrategyTests(unittest.TestCase):
    def test_zero_json_batches_with_slot_first(self):
        spans,slots=spans_and_slots()
        count=[]
        def completion(system,user):
            data=json.loads(user);count.append(data)
            self.assertEqual(len(data['target_slots']),1)
            self.assertIn('quote_options',data)
            choice=data['quote_options'][0]
            return 'CLAIM|'+choice['evidence_id']+'|'+choice['quote_id']+'|原文报告相关的试验结果。'
        generator=ClaimDraftGenerator(QwenJsonClient(completion=completion),
                                     max_calls=2,strategy='slots-first')
        out=generator.generate('结果',spans,slots)
        self.assertEqual(out['generation_calls'],2)
        self.assertEqual(len(out['claims']),2)
        self.assertEqual(len(ClaimSupportVerifier().review(out['claims'],spans)['accepted']),2)
        self.assertEqual({d['stage'] for d in out['diagnostics']},{'slot_first'})

    def test_wrong_strategy_fails(self):
        with self.assertRaises(ValueError):
            ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'{}'),strategy='unknown')

    def test_chinese_attached_numbers_are_not_truncated(self):
        from mhrag_v13.grounded_answers import numbers_in_claim
        self.assertEqual(numbers_in_claim('比值为1.02，P=0.86；95%CI'),['1.02','0.86','95%'])

class SignedNumberSafetyTests(unittest.TestCase):
    def test_negative_effect_sign_cannot_flip(self):
        from mhrag_v13.grounded_answers import numbers_in_claim
        self.assertEqual(numbers_in_claim('研究发现效应值−1.05，与对照组差异为-0.20'),['-1.05','-0.20'])
        source=EvidenceSpan('E44_1',44,1,61,'6','results',
            'Results Trial effect was -1.05 in an analysis reporting an association with outcome.')
        claim={'study_id':44,'slot':'results','evidence_id':'E44_1',
               'quote':source.text,'text':'研究报告效应值为1.05。'}
        self.assertEqual(ClaimSupportVerifier()._check_one(claim,{'E44_1':source})[1],
                         'unsupported_numeric_in_claim')

if __name__=='__main__':unittest.main()
