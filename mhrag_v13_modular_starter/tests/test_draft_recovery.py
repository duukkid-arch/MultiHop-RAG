"""Offline regression tests: bounded micro-batching and observable Qwen failures."""
import json
import unittest
from mhrag_v13.evidence_slots import EvidenceSpan, Slot
from mhrag_v13.grounded_answers import ClaimDraftGenerator, ClaimSupportVerifier, QwenJsonClient

QUOTES={
    (44,'design'):'Methods Study design: BESS was a multicentre blinded randomized parallel-group trial.',
    (44,'primary_outcome'):'Primary outcome was the total duration of mechanical ventilation in hours.',
    (44,'results'):'Results The geometric mean ratio was 1.02 with 95% CI 0.84 to 1.24; p=0.86.',
    (55,'design'):'Methods Study Design was a prospective single-center randomized controlled trial.',
    (55,'primary_outcome'):'Primary outcome included objective neurocognitive attention scores measured using TOVA.',
    (55,'results'):'Results The TOVA attention scores improved in the combined group p=0.006.',
}


def fixtures():
    spans=[];slots=[]
    n=10
    for (did,slot),text in QUOTES.items():
        n+=1
        section='results' if slot=='results' else ('methods' if slot=='design' else 'outcome_definition')
        obj=EvidenceSpan(f'E{did}_{n}',did,n,n,'3',section,text)
        spans.append(obj)
        slots.append(Slot(did,slot,'candidate',(obj.evidence_id,),'candidate'))
    return spans,slots


def valid_claim(data,slot=None,quote_mutator=None):
    d=data['study_id']
    which=slot or data['target_slots'][0]
    candidates=[x for x in data['evidence'] if x['document_id']==d and
        (which!='results' or x['section']=='results') and
        (which!='design' or x['section']=='methods') and
        (which!='primary_outcome' or x['section']=='outcome_definition')]
    if not candidates:
        return []
    s=candidates[0]
    quote=quote_mutator(s['text']) if quote_mutator else s['text']
    # Numberless claim so cite proof, not clinical support, is the focus.
    return [{'study_id':d,'slot':which,'text':'研究报告了相应的原始指标或设计。',
              'evidence_id':s['evidence_id'],'quote':quote}]


class DraftRecoveryTests(unittest.TestCase):
    def test_empty_study_batches_trigger_slot_recovery(self):
        seen=[]
        def completion(sys,user):
            data=json.loads(user);seen.append(data)
            if len(data['target_slots'])>1:return json.dumps({'claims':[],'limitations':['study batch empty']})
            return json.dumps({'claims':valid_claim(data),'limitations':[]})
        spans,slots=fixtures()
        gen=ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=8)
        out=gen.generate('比较设计、主要终点、结果',spans,slots)
        self.assertEqual(out['generation_calls'],8)
        self.assertEqual(len(out['claims']),6)
        self.assertEqual(out['generation_status'],'partial')
        self.assertEqual(out['diagnostics'][0]['status'],'empty_claims')
        self.assertTrue(all(len(d['evidence'])<=6 for d in seen))
        self.assertTrue(all(len({x['document_id'] for x in d['evidence']})==1 for d in seen))
        self.assertEqual(len(ClaimSupportVerifier().review(out['claims'],spans)['accepted']),6)

    def test_batch_success_avoids_extra_calls(self):
        def completion(_,user):
            data=json.loads(user)
            return json.dumps({'claims':[c for sl in data['target_slots']
                                        for c in valid_claim(data,sl)],'limitations':[]})
        spans,slots=fixtures()
        out=ClaimDraftGenerator(QwenJsonClient(completion=completion)).generate('比较设计、终点、结果',spans,slots)
        self.assertEqual(out['generation_calls'],2)
        self.assertEqual(out['generation_status'],'ok')
        self.assertEqual(len(out['claims']),6)

    def test_invalid_json_first_then_slot_recovery(self):
        def completion(_,user):
            data=json.loads(user)
            if len(data['target_slots'])>1:return 'unparseable model output'
            return json.dumps({'claims':valid_claim(data),'limitations':[]})
        spans,slots=fixtures()
        out=ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=4).generate('比较设计、终点、结果',spans,slots)
        self.assertEqual(out['generation_calls'],4)
        self.assertEqual(out['diagnostics'][0]['status'],'model_error')
        self.assertTrue(out['claims'])
        self.assertTrue(any(v['stage']=='slot_retry' for v in out['diagnostics']))

    def test_fabricated_numeric_quote_rejected_and_retried(self):
        calls=[]
        def completion(_,user):
            data=json.loads(user);calls.append(data)
            if len(data['target_slots'])>1:
                return json.dumps({'claims':valid_claim(data,'results',lambda t:t+' invented appendix'), 'limitations':[]})
            return json.dumps({'claims':valid_claim(data),'limitations':[]})
        spans,slots=fixtures()
        out=ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=4).generate('比较设计、终点、结果',spans,slots)
        self.assertTrue(any(x['status']=='all_failed_precheck' for x in out['diagnostics']))
        self.assertTrue(any(x['status']=='drafted' for x in out['diagnostics']))
        review=ClaimSupportVerifier().review(out['claims'],spans)
        self.assertTrue(any(x['check']=='quote_not_in_original_chunk' for x in review['rejected']))

    def test_max_one_call_budget_no_invention(self):
        spans,slots=fixtures()
        out=ClaimDraftGenerator(QwenJsonClient(completion=lambda *_:'{"claims":[],"limitations":[]}'),max_calls=1).generate('比较设计、终点、结果',spans,slots)
        self.assertEqual(out['generation_calls'],1)
        self.assertEqual(len(out['claims']),0)
        self.assertIn('draft_call_budget_exhausted',out['limitations'])

    def test_model_unavailable_never_fabricates(self):
        def completion(*args):raise OSError('offline')
        spans,slots=fixtures()
        out=ClaimDraftGenerator(QwenJsonClient(completion=completion),max_calls=2).generate('比较设计、终点、结果',spans,slots)
        self.assertEqual(len(out['claims']),0)
        self.assertEqual(out['generation_status'],'failed')
        self.assertTrue(all(x['error_type']=='OSError' for x in out['diagnostics']))

    def test_excludes_unconfirmed_study_evidence(self):
        spans,slots=fixtures()
        slots=[s for s in slots if s.document_id==44]
        out=ClaimDraftGenerator(QwenJsonClient(completion=lambda s,u:'{"claims":[],"limitations":[]}'),max_calls=1).generate('设计',spans,slots)
        self.assertEqual({x['document_id'] for x in out['diagnostics']},{44})

    def test_configuration_limits(self):
        c=QwenJsonClient(completion=lambda *_:'{}')
        for calls in (-1,0,13):
            with self.assertRaises(ValueError):ClaimDraftGenerator(c,max_calls=calls)
        with self.assertRaises(ValueError):ClaimDraftGenerator(c,max_excerpt_chars=100)

if __name__=='__main__':unittest.main()
