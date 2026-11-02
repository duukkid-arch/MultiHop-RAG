"""Prevent the live v0.4 problem: poor quote selection + multi-study refusal."""
import json
import unittest
from mhrag_v13.evidence_slots import EvidenceSpan,Slot
from mhrag_v13.grounded_answers import ClaimDraftGenerator,QwenJsonClient,ClaimSupportVerifier


def specimen():
    a=EvidenceSpan('E44_4882',44,4882,52,'5','results',
        'Results From Dec 18, 2018, to March 31, 2024, 1009 infants were assessed for eligibility. '
        +'A '*220+'The geometric mean ratio of the endotracheal poractant alfa group '
        'compared with sham was 1.02 (95% CI 0.84 to 1.24; p=0.86).')
    b=EvidenceSpan('E55_6062',55,6062,47,'7','results',
        'Results Baseline Characteristics and intention to treat analysis. '+'B '*190+
        'The combination group demonstrated greater gains in TOVA attention comparison '
        'scores at week eight (p=0.006, Cohen d=0.809).')
    m=EvidenceSpan('E44_4848',44,4848,20,'2','methods',
        'Methods We used regression for statistical analysis. '+'C '*190+
        'The BESS was a multicentre, blinded, randomised, sham-controlled, '
        'parallel-group phase 2 superiority trial.')
    o=EvidenceSpan('E44_4867',44,4867,40,'4','outcome_definition',
        'The primary outcome analysis excluded withdrawals. '+'D '*190+
        'The primary outcome was total duration of invasive mechanical ventilation in hours.')
    return a,b,m,o


class EvidenceWindowV05(unittest.TestCase):
    def _gen(self, completion=lambda *_:'NO_EVIDENCE'):
        return ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=6,strategy='text-first')

    def test_results_use_numeric_window_not_study_flow(self):
        a,b,_,_=specimen()
        for span, phrase in [(a,'1.02'),(b,'0.006')]:
            w=self._gen()._text_first_choices([span],'results')[0]
            self.assertIn(phrase,w['quote'])
            self.assertIn(w['quote'],span.text)

    def test_design_uses_trial_design_not_stats_prequel(self):
        _,_,m,_=specimen()
        w=self._gen()._text_first_choices([m],'design')[0]
        self.assertIn('multicentre',w['quote'])
        self.assertIn('randomised',w['quote'])

    def test_primary_prefers_definition_over_analysis(self):
        _,_,_,o=specimen()
        w=self._gen()._text_first_choices([o],'primary_outcome')[0]
        self.assertIn('total duration',w['quote'])

    def test_one_study_question_removed_from_model_prompt(self):
        a,_,_,_=specimen()
        slots=[Slot(44,'results','candidate',(a.evidence_id,),'candidate')]
        def completion(system,user):
            req=json.loads(user)
            self.assertNotIn('question',req)
            self.assertIn('1.02',req['original_evidence_quote'])
            self.assertIn('绝不比较两篇论文',system)
            return '本研究结果报告几何均值比为1.02，p=0.86。'
        r=self._gen(completion).generate('对比BESS和Focus Pro的疗效', [a],slots)
        self.assertEqual(len(r['claims']),1)
        self.assertEqual(len(ClaimSupportVerifier().review(r['claims'],[a])['accepted']),1)
        self.assertEqual(r['diagnostics'][0]['chunk_id'],4882)
        self.assertIn('1.02',r['diagnostics'][0]['selected_quote_preview'])

    def test_declines_are_not_coerced_into_claims(self):
        a,_,_,_=specimen()
        slots=[Slot(44,'results','candidate',(a.evidence_id,),'candidate')]
        r=self._gen().generate('结果',[a],slots)
        self.assertEqual(r['claims'],[])
        self.assertEqual(r['diagnostics'][0]['status'],'empty_claims')
        self.assertIn('selected_quote_preview',r['diagnostics'][0])

    def test_false_numeric_cannot_pass(self):
        a,_,_,_=specimen()
        slots=[Slot(44,'results','candidate',(a.evidence_id,),'candidate')]
        r=self._gen(lambda *_:'本研究结果几何均值比为9.9。').generate('结果',[a],slots)
        self.assertEqual(len(r['claims']),1)
        self.assertEqual(ClaimSupportVerifier().review(r['claims'],[a])['accepted'],[])

    def test_preview_is_pure_and_has_source_provenance(self):
        a,b,m,o=specimen()
        slots=[Slot(44,'design','candidate',(m.evidence_id,),'candidate'),
               Slot(44,'primary_outcome','candidate',(o.evidence_id,),'candidate'),
               Slot(55,'results','candidate',(b.evidence_id,),'candidate')]
        rows=self._gen().preview_evidence_windows([a,b,m,o],slots)
        self.assertEqual(len(rows),3)
        for row in rows:
            self.assertEqual(row['selected']['document_id'],row['document_id'])
            self.assertGreater(len(row['selected']['quote']),34)

    def test_unknown_slot_blocked(self):
        with self.assertRaises(ValueError):self._gen()._text_first_choices([],'discussions')

if __name__=='__main__':unittest.main()
