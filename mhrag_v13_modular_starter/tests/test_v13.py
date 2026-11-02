import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mhrag_v13 import AdaptiveController,Source,Study,Resolution,SearchResult,Registry
from mhrag_v13.verify import StructuralEvidenceGate,unique_sources
from mhrag_v13.resolve import SQLiteStudyResolver,anchors
from mhrag_v13.answer import EvidencePacketAnswerer
from mhrag_v13.retrieve import LocalChatRetriever
from mhrag_v13.qwen_answer import LocalQwenAnswerer
from mhrag_v13.open_document import SQLiteSectionReader
from mhrag_v13.tracing import NullTrace

LONG='Results: Study effect was reported in the Results section with control and intervention comparisons.'
def source(did,chunk=1,snippet=LONG):
    return Source(did,chunk,'3',f'paper-{did}.pdf',snippet)
class FakeResolver:
    def __init__(self,resolution):self.resolution=resolution
    def resolve(self,question):return self.resolution
class FakeRetriever:
    def __init__(self, answers):self.answers=answers;self.calls=[]
    def search(self,question,*,agentic=False,document_ids=None):
        self.calls.append((agentic,tuple(document_ids) if document_ids is not None else None))
        key=('agent' if agentic else 'scope' if document_ids is not None else 'rag')
        return SearchResult(list(self.answers.get((key,tuple(document_ids or [])),self.answers.get(key,[]))),elapsed_ms=10)
class TestCore(unittest.TestCase):
    def setUp(self):
        self.gate=StructuralEvidenceGate();self.answer=EvidencePacketAnswerer()
        self.res=Resolution((Study(44,'BESS.pdf','BESS'),Study(55,'Focus-Pro.pdf','Focus Pro')),(),('Focus Pro','BESS'))
    def controller(self,fetch,res=None,budget=4):
        return AdaptiveController(FakeResolver(res or self.res),fetch,self.gate,self.answer,
                                  trace=NullTrace(),max_http_calls=budget)
    def test_broad_fully_covered_no_agent_or_repairs(self):
        f=FakeRetriever({'rag':[source(44),source(55)]})
        out=self.controller(f).run('比较 Focus Pro 和 BESS 的主要结果')
        self.assertEqual(out['http_calls'],1)
        self.assertEqual(out['assessment'],'STUDIES_LOCATED_FACTS_UNVERIFIED')
    def test_targeted_gap_repair(self):
        f=FakeRetriever({'rag':[source(44)],('scope',(55,)):[source(55)]})
        out=self.controller(f).run('比较 Focus Pro 和 BESS 的主要结果')
        self.assertEqual(out['observed_ids'],[44,55]);self.assertEqual(out['http_calls'],2)
        self.assertIn((False,(55,)),f.calls)
    def test_no_gold_used_in_controller(self):
        f=FakeRetriever({'rag':[source(44)],('scope',(55,)):[source(55)]})
        self.controller(f).run('比较 Focus Pro 和 BESS 的主要结果')
        self.assertTrue(all(d is None or d==(55,) for _,d in f.calls))
    def test_missing_study_never_claims_complete(self):
        f=FakeRetriever({'rag':[source(44)]})
        out=self.controller(f,budget=2).run('比较 Focus Pro 和 BESS 的结果')
        self.assertIn(out['assessment'],('MISSING_REQUIRED_STUDIES',))
        self.assertIn(55,out['missing_ids']);self.assertFalse(out['claim_facts_verified'])
    def test_budget_hard_cap(self):
        f=FakeRetriever({'rag':[source(44)]})
        out=self.controller(f,budget=1).run('比较 Focus Pro 和 BESS 的结果')
        self.assertEqual(len(f.calls),1)
    def test_undefined_entity_not_scoped(self):
        r=Resolution((),('unknown_named_study',),('unknown_named_study',))
        f=FakeRetriever({'rag':[source(44)],'agent':[source(44)]})
        out=self.controller(f,res=r).run('比较研究 A 和 B 的疗效')
        self.assertEqual(out['assessment'],'UNRESOLVED_STUDY_IDENTITY')
        self.assertTrue(all(scope is None for _,scope in f.calls))
    def test_malformed_api_source_rejected(self):
        with self.assertRaises(ValueError):Source.from_api({'document_id':'44'})
    def test_source_dedup(self):
        self.assertEqual(len(unique_sources([source(44),source(44),source(55)])),2)
    def test_reference_only_guard(self):
        a=self.gate.assess(Resolution((Study(44,'paper','F'),),(),('F',)),
                           [source(44,snippet='References\n[1] Some cited clinical trial')],'q')
        self.assertEqual(a.status,'REFERENCE_ONLY_OR_THIN_EVIDENCE')
    def test_claim_verification_always_false(self):
        a=self.gate.assess(self.res,[source(44),source(55)],'compare')
        self.assertFalse(a.claim_facts_verified)
    def test_no_sources_abstain(self):
        a=self.gate.assess(Resolution(),[],'q')
        self.assertIn('没有找到',self.answer.compose('q',[],a))
    def test_plugin_registration(self):
        r=Registry();r.register('retriever','fake',FakeRetriever)
        self.assertIsInstance(r.create('retriever','fake',answers={}),FakeRetriever)
        with self.assertRaises(ValueError):r.register('retriever','fake',FakeRetriever)
    def test_no_external_http_by_default(self):
        with self.assertRaises(ValueError):LocalChatRetriever('https://external.example.com:443')
    def test_bad_docid_in_scope(self):
        r=LocalChatRetriever()
        with self.assertRaises(ValueError):r.search('hi',document_ids=[-1])
    def test_bad_http_budget(self):
        with self.assertRaises(ValueError):self.controller(FakeRetriever({}),budget=0)
    def test_llm_falls_back_when_study_missing(self):
        a=self.gate.assess(self.res,[source(44)],'q')
        result=LocalQwenAnswerer().compose('q',[source(44)],a)
        self.assertIn('55',result)
    def test_sqlite_reader_and_resolver(self):
        with tempfile.TemporaryDirectory() as root:
            db=Path(root)/'knowledge.sqlite'
            con=sqlite3.connect(db)
            con.executescript('CREATE TABLE documents(id INTEGER, filename TEXT);'
                   'CREATE TABLE chunks(id INTEGER,document_id INTEGER,ordinal INTEGER,page INTEGER,body TEXT);')
            con.execute('INSERT INTO documents VALUES(44,?)',('BESS clinical study.pdf',))
            con.execute('INSERT INTO documents VALUES(55,?)',('Focus Pro trial.pdf',))
            con.execute('INSERT INTO chunks VALUES(1,44,0,1,?)',('BESS clinical study introduction '+LONG,))
            con.execute('INSERT INTO chunks VALUES(2,55,0,2,?)',('Focus Pro trial '+LONG,))
            con.commit();con.close()
            rr=SQLiteStudyResolver(db).resolve('比较 Focus Pro 与 BESS 的结果')
            self.assertEqual(rr.ids,(44,55));self.assertTrue(rr.complete)
            result=SQLiteSectionReader(db).open(55,'BESS 结果')
            self.assertEqual(result[0].origin,'local_open')
    def test_reviewed_alias_resolves_metadata_ambiguity(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            db=Path(root)/'knowledge.sqlite'
            con=sqlite3.connect(db)
            con.executescript('CREATE TABLE documents(id INTEGER, filename TEXT);'
                  'CREATE TABLE chunks(id INTEGER,document_id INTEGER,ordinal INTEGER,page INTEGER,body TEXT);')
            docs=[(44,'BESS original.pdf','BESS is the prospective trial.'),
                  (55,'focus-therapy.pdf','This original study tested Focus Pro.'),
                  (66,'commentary.pdf','Focus Pro was discussed by another review.')]
            for did,name,body in docs:
                con.execute('INSERT INTO documents VALUES (?,?)',(did,name))
                con.execute('INSERT INTO chunks VALUES(?,?,?,?,?)',(did,did,0,1,body))
            con.commit();con.close()
            q='比较 Focus Pro 和 BESS 两项研究的设计'
            self.assertIn('Focus Pro',SQLiteStudyResolver(db).resolve(q).unresolved)
            manifest=Path(root)/'aliases.json'
            def save(verified=True,wrong=False):
                manifest.write_text(json.dumps({'aliases':[
                    {'alias':'Focus Pro','document_id':44 if wrong else 55,
                     'source_phrase':'Focus Pro', 'verified_against_source':verified},
                    {'alias':'BESS','document_id':44,
                     'source_phrase':'BESS','verified_against_source':verified}]}),encoding='utf-8')
            save(False)
            self.assertIn('Focus Pro',SQLiteStudyResolver(db,manifest).resolve(q).unresolved)
            save(True)
            r=SQLiteStudyResolver(db,manifest).resolve(q)
            self.assertEqual(r.ids,(44,55));self.assertTrue(r.complete)
            save(True,True)
            with self.assertRaises(ValueError):SQLiteStudyResolver(db,manifest)

    def test_ambiguous_title_unresolved(self):
        with tempfile.TemporaryDirectory() as root:
            db=Path(root)/'knowledge.sqlite';con=sqlite3.connect(db)
            con.executescript('CREATE TABLE documents(id INTEGER, filename TEXT);'
                  'CREATE TABLE chunks(id INTEGER,document_id INTEGER,ordinal INTEGER,page INTEGER,body TEXT);')
            for i in (1,2):
                con.execute('INSERT INTO documents VALUES (?,?)',(i,'BESS trial.pdf'))
            con.commit();con.close()
            rr=SQLiteStudyResolver(db).resolve('BESS trial')
            self.assertIn('BESS',rr.unresolved)

if __name__=='__main__':unittest.main()
