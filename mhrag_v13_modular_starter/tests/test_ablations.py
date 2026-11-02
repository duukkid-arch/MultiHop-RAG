"""Offline regression for real independent E1/E2/E3 implementation.
All clinical statements used here are synthetic test fixtures, not medical facts.
"""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mhrag_v13.ablation_arms import (
    decompose_question,QuestionAwareHarvester,QuestionAwareGenerator,
    e3_safety_review,AblationEngine,_is_focused_claim,_target,
)
from mhrag_v13.evidence_slots import EvidenceSpan,Slot
from benchmarks.v13_ablations_runner import make_plan,ARMS,summarize
from mhrag_v13.grounded_answers import QwenJsonClient


class FixtureDB(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=Path(self.tmp.name)/'knowledge.sqlite'
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY)')
            c.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INT, ordinal INT, page TEXT,body TEXT)')
            c.execute('INSERT INTO documents(id) VALUES(44)')
            rows=[
                (1,44,0,'1','Abstract. This controlled trial studies invented DrugX in adults. '+ 'A'*80),
                (2,44,12,'3','Methods\nParticipants aged 18-65 years were enrolled at 4 hospitals; initial dose 5 mg/kg daily for 14 days. '+ 'C'*65),
                (3,44,26,'5','Results\nAt 8 weeks response was 24% versus 19% in the placebo group; p=0.45. '+ 'D'*65),
                (4,44,45,'9','References\nSmith et al. In a separate reference: all patients cured with dose 200 mg/kg in 2030. '+ 'E'*75),
                (5,44,46,'10','Other study from references says children were aged 3-5 years and 25% died. '+ 'F'*75),
            ]
            c.executemany('INSERT INTO chunks(id,document_id,ordinal,page,body) VALUES(?,?,?,?,?)',rows)

    def tearDown(self):self.tmp.cleanup()

    def test_subtargets_specific(self):
        self.assertEqual([t.name for t in decompose_question('GASH拟纳入患者的年龄与发病时间要求是什么？')],['age','onset'])
        self.assertIn('dose',[t.name for t in decompose_question('给药剂量和疗程如何设置？')])
        self.assertIn('duration',[t.name for t in decompose_question('给药剂量和疗程如何设置？')])
        self.assertIn('date',[t.name for t in decompose_question('研讨会何时举办？')])

    def test_reader_finds_question_and_ignores_references(self):
        spans,slots=QuestionAwareHarvester(self.db).harvest([44],'DrugX首次给药剂量和疗程如何设置？')
        self.assertTrue(spans)
        self.assertTrue(any('5 mg/kg' in s.text for s in spans))
        self.assertFalse(any('200 mg/kg' in s.text for s in spans))
        self.assertFalse(any(s.chunk_id in (4,5) for s in spans))
        self.assertTrue(any(slot.status=='candidate' for slot in slots))

    def test_reader_readonly(self):
        before=self.db.read_bytes()
        QuestionAwareHarvester(self.db).harvest([44],'给药剂量多少？')
        self.assertEqual(before,self.db.read_bytes())

    def test_reader_reject_wrong_document_id(self):
        with self.assertRaises(ValueError):QuestionAwareHarvester(self.db).harvest([True],'给药剂量')

    def test_source_provenance(self):
        spans,_=QuestionAwareHarvester(self.db).harvest([44],'给药剂量和疗程？')
        self.assertTrue(all(s.document_id==44 and s.evidence_id==f'E44_{s.chunk_id}' for s in spans))

    def test_no_generic_study_type_as_dosage(self):
        self.assertFalse(_is_focused_claim('这是一项随机对照试验。',_target('dose')))
        self.assertFalse(_is_focused_claim('主要讨论了候选药物。',_target('age')))
        self.assertTrue(_is_focused_claim('首次剂量为5 mg/kg。',_target('dose')))

    def test_targeted_generator_valid_numeric_quote(self):
        spans,slots=QuestionAwareHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        client=QwenJsonClient(completion=lambda system,user:'首次给药剂量为5 mg/kg。')
        result=QuestionAwareGenerator(client,max_calls=3).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertEqual(len(result['claims']),1)
        self.assertEqual(result['claims'][0]['study_id'],44)
        self.assertTrue(result['claims'][0]['quote'] in next(s.text for s in spans if s.evidence_id==result['claims'][0]['evidence_id']))

    def test_numeric_hallucination_blocked(self):
        spans,slots=QuestionAwareHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        client=QwenJsonClient(completion=lambda system,user:'首次给药剂量为999 mg/kg。')
        result=QuestionAwareGenerator(client,max_calls=3).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertFalse(result['claims'])
        self.assertTrue(any(d['status']=='rejected_quote_numeric' for d in result['diagnostics']))

    def test_no_evidence(self):
        spans,slots=QuestionAwareHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        client=QwenJsonClient(completion=lambda system,user:'NO_EVIDENCE')
        result=QuestionAwareGenerator(client,max_calls=3).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertFalse(result['claims'])
        self.assertEqual(result['generation_status'],'empty')

    def test_e3_no_claim_abstain(self):
        result={'claims':{'accepted':[]},'identity':{'required_ids':[44],'complete':True,'unresolved':[]},
                'routing':{'mode':'confirmed_named'},'generation_diagnostics':[]}
        out=e3_safety_review(result,'首次给药剂量是多少？')
        self.assertEqual(out['decision']['decision'],'abstain')
        self.assertEqual(out['ablation_safety']['status'],'abstain')

    def test_e3_universal_warning(self):
        result={'claims':{'accepted':[]},'identity':{'required_ids':[44],'complete':True,'unresolved':[]},
                'routing':{'mode':'confirmed_named'},'generation_diagnostics':[]}
        out=e3_safety_review(result,'能否证明所有患者都获益？')
        self.assertIn('universal_or_absolute_claim_requires_human_review',out['ablation_safety']['notes'])

    def test_e3_unverified_identity(self):
        result={'claims':{'accepted':[]},'identity':{'required_ids':[],'complete':False,'unresolved':[]},
                'routing':{'mode':'provisional_api'},'generation_diagnostics':[]}
        out=e3_safety_review(result,'首次给药剂量是多少？')
        self.assertIn('unconfirmed_study_identity',out['ablation_safety']['notes'])

    def test_four_arms_randomized_and_complete(self):
        cases=[{'case_id':'a'},{'case_id':'b'}]
        p=make_plan(cases,42)
        self.assertEqual(len(p),8)
        for cid in ('a','b'):
            self.assertEqual({x['arm'] for x in p if x['case_id']==cid},set(ARMS))
        self.assertEqual(p,make_plan(cases,42))

    def test_summary_does_not_invent_accuracy(self):
        plan=make_plan([{'case_id':'a'}],42)
        rows=[{'case_id':'a','arm':p['arm'],'status':'ok','wall_ms':10,'accepted_claim_count':0,
                'decision':{'decision':'abstain'}} for p in plan]
        s=summarize(rows,plan)
        self.assertEqual(s['medical_accuracy'],'NOT_EVALUATED')
        self.assertEqual(s['all_four_arms_successful_cases'],1)

    def test_invalid_arm(self):
        engine=AblationEngine(resolver=None,retriever=None,db_path=self.db,llm=None)
        with self.assertRaises(ValueError):engine.run('G','sample question')


if __name__=='__main__':unittest.main()

class EndToEndAblationIntegration(unittest.TestCase):
    def test_all_three_arms_really_execute_source_and_generation(self):
        from mhrag_v13.contracts import Resolution,Study,Source,SearchResult
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'fixture.sqlite'
            with sqlite3.connect(db) as c:
                c.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY)')
                c.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INT,ordinal INT,page TEXT,body TEXT)')
                c.execute('INSERT INTO documents VALUES(44)')
                c.executemany('INSERT INTO chunks VALUES(?,?,?,?,?)',[
                    (1,44,1,'1','Abstract. This synthetic trial concerns invented DrugX. '+'A'*90),
                    (2,44,12,'3','Methods\nIn this fabricated trial the initial DrugX dose was 5 mg/kg. '+'B'*90),
                    (3,44,25,'6','Results\nAt end of trial the response was 10% versus 9%. '+'C'*90),
                    (4,44,40,'8','References\nThe entirely unrelated paper tested 999 mg/kg. '+'D'*90),
                ])
            class Resolver:
                def resolve(self,q):return Resolution(studies=(Study(44,'Synthetic Trial','DrugX'),),anchors=('DrugX',))
            class Retriever:
                def __init__(self):self.calls=0
                def search(self,q,*,agentic=False,document_ids=None):
                    self.calls+=1
                    return SearchResult([Source(44,2,'3','Synthetic Trial','Methods initial dose 5 mg/kg. '+'Z'*70)],'fallback',80)
            llm=QwenJsonClient(completion=lambda sys,prompt:
                '首次剂量为5 mg/kg。' if 'this_subquestion' in prompt else 'NO_EVIDENCE')
            retriever=Retriever()
            engine=AblationEngine(resolver=Resolver(),retriever=retriever,db_path=db,llm=llm,max_calls=2)
            answers={a:engine.run(a,'DrugX首次给药剂量是多少？') for a in ('E1','E2','E3')}
            self.assertEqual(set(answers),{'E1','E2','E3'})
            self.assertEqual(answers['E1']['accepted_claim_count'],0)
            self.assertGreaterEqual(answers['E2']['accepted_claim_count'],1)
            self.assertGreaterEqual(answers['E3']['accepted_claim_count'],1)
            self.assertEqual(answers['E3']['ablation_arm'],'E3')
            self.assertTrue(answers['E3']['question_coverage'])
            self.assertFalse(answers['E3']['medical_claims_verified'])
            self.assertGreaterEqual(retriever.calls,3)

class ReviewerExportTests(unittest.TestCase):
    def test_refuses_incomplete_run(self):
        from benchmarks.v13_ablations_review import export_reviews
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';run.mkdir()
            (run/'manifest.json').write_text(json.dumps({'kind':'historical_dev_91_not_holdout_NOT_MEDICALLY_VALIDATED','config':{'cases':'none'}}))
            (run/'summary.json').write_text(json.dumps({'state':'time_cap','completed':12,'total_runtime_errors':0}))
            with self.assertRaises(ValueError):export_reviews(run)

    def test_export_counts_and_private_mapping(self):
        from benchmarks.v13_ablations_review import export_reviews
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            cases=root/'cases.json'
            cases.write_text(json.dumps({'cases':[{'case_id':f'c{i}','question':f'question{i}'} for i in range(91)]}))
            (run/'manifest.json').write_text(json.dumps({'kind':'historical_dev_91_not_holdout_NOT_MEDICALLY_VALIDATED','config':{'cases':str(cases)}}))
            (run/'summary.json').write_text(json.dumps({'state':'complete','completed':364,'total_runtime_errors':0}))
            with (run/'results.jsonl').open('w') as f:
                for i in range(91):
                    for arm in ARMS:
                        f.write(json.dumps({'case_id':f'c{i}','arm':arm,'answer':arm,'status':'ok'})+'\n')
            out,count=export_reviews(run)
            self.assertEqual(count,364)
            with (out/'blinded_R1_364_answers.csv').open(encoding='utf-8-sig') as f:
                import csv
                data=list(csv.DictReader(f))
            self.assertEqual(len(data),364)
            self.assertNotIn('arm',data[0])
            self.assertTrue((out/'ADMIN_UNBLIND_AFTER_ADJUDICATION.csv').is_file())
            with self.assertRaises(ValueError):export_reviews(run)
