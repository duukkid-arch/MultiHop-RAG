import json,sqlite3,tempfile,unittest
from pathlib import Path
from mhrag_v13.e2_repair_trial import (repair_targets,RepairHarvester,RepairGenerator,
    RepairEngine,posthoc_e3_from_repaired)
from mhrag_v13.grounded_answers import QwenJsonClient
from benchmarks.run_e2_repair16 import load_cases,plan_for,make_summary


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.db=Path(self.t.name)/'sample.sqlite'
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY)')
            db.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INT,ordinal INT,page TEXT,body TEXT)')
            db.execute('INSERT INTO documents VALUES(44)')
            db.executemany('INSERT INTO chunks VALUES(?,?,?,?,?)',[
                (1,44,1,'1','Abstract. This experiment studied DrugX in adults and report summary. '+'A'*120),
                (2,44,12,'3','Methods\nThe intervention involved DrugX initial dose 5 mg/kg. '+'B'*120),
                (3,44,17,'4','Methods continued. DrugX was administered for 14 days and followed up at week 12. '+'C'*120),
                (4,44,28,'6','Results\nFor the treatment group OR 0.72 (95% CI 0.61-0.85), p=0.01 was reported. '+'D'*120),
                (5,44,40,'8','References\nUnrelated article claims OR 999 and dose 800 mg/kg. '+'E'*120),
                (6,44,41,'9','Another cited trial says OR 444 and dose 777 mg/kg. '+'F'*120),
            ])
    def tearDown(self):self.t.cleanup()

    def test_intent_or(self):
        self.assertIn('effect_estimate',[t.name for t in repair_targets('总屏幕时间每增加1小时，数学高成绩的OR是多少？')])
    def test_literature_count(self):
        self.assertEqual(repair_targets('综述共纳入多少文章和独立随机试验？')[0].name,'literature_count')
    def test_absolute_evidence(self):
        self.assertEqual(repair_targets('是否证明所有患者必然获益？')[0].name,'evidence_limit')
    def test_followup_time(self):
        self.assertEqual(repair_targets('第12周为何时间节点？')[0].name,'followup_time')
    def test_dose_and_duration(self):
        self.assertEqual([t.name for t in repair_targets('给药剂量和疗程如何设置？')],['dose','duration'])
    def test_source_refs_excluded_and_late_methods(self):
        spans,_=RepairHarvester(self.db).harvest([44],'DrugX剂量和疗程多久？')
        self.assertTrue(any('5 mg/kg' in s.text for s in spans))
        self.assertTrue(any('14 days' in s.text for s in spans))
        self.assertFalse(any(s.chunk_id in (5,6) for s in spans))
        self.assertTrue(all(s.evidence_id==f'E44_{s.chunk_id}' for s in spans))
    def test_or_source_ranked(self):
        spans,_=RepairHarvester(self.db).harvest([44],'该试验OR与置信区间是多少？')
        self.assertTrue(any('OR 0.72' in s.text for s in spans))
    def test_sqlite_readonly(self):
        before=self.db.read_bytes()
        RepairHarvester(self.db).harvest([44],'DrugX剂量多少？')
        self.assertEqual(before,self.db.read_bytes())
    def test_numeric_good_claim(self):
        spans,slots=RepairHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        model=QwenJsonClient(completion=lambda _s,_u:'首次给药剂量为5 mg/kg。' if '5 mg/kg' in _u else 'NO_EVIDENCE')
        gen=RepairGenerator(model,max_calls=6).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertEqual(len(gen['claims']),1)
        self.assertTrue(any(d['status']=='supported_quote_numeric_precheck' for d in gen['diagnostics']))
    def test_numeric_hallucination_rejected(self):
        spans,slots=RepairHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        model=QwenJsonClient(completion=lambda _s,_u:'首次给药剂量为999 mg/kg。')
        gen=RepairGenerator(model,max_calls=6).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertEqual(gen['claims'],[])
        self.assertTrue(any(d['status']=='rejected_quote_numeric' for d in gen['diagnostics']))
    def test_fallback_second_chunk(self):
        spans,slots=RepairHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        calls=[]
        def completion(_s,u):
            calls.append(json.loads(u)['source_excerpt'])
            return 'NO_EVIDENCE' if len(calls)==1 else '首次给药剂量为5 mg/kg。' if '5 mg/kg' in u else 'NO_EVIDENCE'
        gen=RepairGenerator(QwenJsonClient(completion=completion),max_calls=6).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertGreaterEqual(len(calls),2)
        self.assertLessEqual(len(calls),6)
    def test_no_evidence_is_not_success(self):
        spans,slots=RepairHarvester(self.db).harvest([44],'DrugX首次给药剂量是多少？')
        gen=RepairGenerator(QwenJsonClient(completion=lambda _s,_u:'NO_EVIDENCE'),max_calls=6).generate('DrugX首次给药剂量是多少？',spans,slots)
        self.assertEqual(gen['generation_status'],'empty')
    def test_posthoc_e3_never_model_call(self):
        r={'claims':{'accepted':[]},'identity':{'required_ids':[44],'complete':True},
           'routing':{'mode':'confirmed_named'},'generation_diagnostics':[]}
        e3=posthoc_e3_from_repaired(r,'是否证明每位患者都获益？')
        self.assertEqual(e3['decision']['decision'],'abstain')
        self.assertIn('universal_or_absolute_claim_requires_human_review',e3['ablation_safety']['notes'])
    def test_plan(self):
        cases=[{'case_id':'a'},{'case_id':'b'}]
        self.assertEqual(len(plan_for(cases,42)),4)
        self.assertEqual(plan_for(cases,42),plan_for(cases,42))
    def test_summary(self):
        plan=[{'case_id':'a','arm':'E2_legacy'},{'case_id':'a','arm':'E2_repaired'}]
        rows=[{'case_id':'a','arm':arm,'status':'ok','wall_ms':100,'accepted_claim_count':0,
               'generation_diagnostics':[{'status':'no_evidence'}]} for arm in ('E2_legacy','E2_repaired')]
        s=make_summary(rows,plan)
        self.assertEqual(s['paired_cases_completed'],1)
        self.assertEqual(s['medical_accuracy'],'NOT_EVALUATED')
    def test_gold_refusal(self):
        obj={'dataset_kind':'historical_dev_16_regression_no_gold','cases':[{'case_id':str(i),'question':'q','gold_id':23} for i in range(16)]}
        p=Path(self.t.name)/'bad.json';p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):load_cases(p)

if __name__=='__main__':unittest.main()

class RepairEndToEndTests(unittest.TestCase):
    def test_real_controller_mock_services_and_non_mutation(self):
        from mhrag_v13.contracts import Resolution,Study,Source,SearchResult
        with tempfile.TemporaryDirectory() as t:
            db=Path(t)/'e2.sqlite'
            with sqlite3.connect(db) as c:
                c.execute('CREATE TABLE documents(id INTEGER PRIMARY KEY)')
                c.execute('CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INT,ordinal INT,page TEXT,body TEXT)')
                c.execute('INSERT INTO documents VALUES(44)')
                c.executemany('INSERT INTO chunks VALUES(?,?,?,?,?)',[
                    (1,44,0,'1','Abstract. A fictional DrugX investigation in patients. '+'X'*100),
                    (2,44,16,'4','Methods\nFictional DrugX initial dose was 5 mg/kg for each person. '+'Y'*100),
                    (3,44,30,'7','References\nA different study mentions 987 mg/kg. '+'Z'*100),
                ])
            before=db.read_bytes()
            class Resolver:
                def resolve(self,_q):return Resolution(studies=(Study(44,'DrugX trial','DrugX'),),anchors=('DrugX',))
            class Retriever:
                def search(self,_q,*,agentic=False,document_ids=None):
                    return SearchResult([Source(44,2,'4','DrugX trial','Methods: 5 mg/kg. '+'A'*90)],'fallback',90)
            client=QwenJsonClient(completion=lambda _system,user:
                '首次剂量为5 mg/kg。' if '5 mg/kg' in json.loads(user).get('source_excerpt','') else 'NO_EVIDENCE')
            result=RepairEngine(resolver=Resolver(),retriever=Retriever(),db_path=db,llm=client).run('DrugX首次给药剂量是多少？')
            self.assertGreaterEqual(result['accepted_claim_count'],1)
            self.assertTrue(any(c['claim']['evidence_id'].startswith('E44_') for c in result['claims']['accepted']))
            self.assertEqual(db.read_bytes(),before)
            self.assertFalse(result['medical_claims_verified'])
