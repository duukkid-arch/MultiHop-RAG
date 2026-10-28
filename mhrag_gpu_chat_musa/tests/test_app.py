from __future__ import annotations
import io,os,tempfile,unittest,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from mhrag_gpu_chat_musa import store,parser
from mhrag_gpu_chat_musa.core import ChatCore

class FakeGPU:
    def __init__(self):self.calls=0
    def embed(self,texts):
        self.calls+=1
        data=[]
        for t in texts:
            v=np.zeros(1024,dtype=np.float32)
            v[0]=1 if any(k in t.casefold() for k in ('clozapine','氯氮平')) else 0
            v[1]=1 if any(k in t.casefold() for k in ('surgery','tavr','心脏')) else 0
            if v.sum()==0:v[2]=1
            v/=np.linalg.norm(v)
            data.append(v)
        return np.vstack(data)
    def score(self,pairs):return [3.0 if 'clozapine' in d.lower() else .2 for q,d in pairs]

class Smoke(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.core=ChatCore(Path(self.temp.name)/'app.sqlite',FakeGPU())
    def tearDown(self):self.temp.cleanup()
    def test_ingest_and_cross_language(self):
        t=('Clozapine in treatment resistant schizophrenia. This study describes comparative efficacy '
           'and important adverse effects with monitoring requirements. ')*3
        doc=self.core.ingest('schizophrenia.md',t.encode())
        self.assertGreater(doc['chunks'],0)
        sources,trace=self.core.retrieve('氯氮平治疗精神分裂症的疗效','hybrid_ce')
        self.assertEqual(sources[0]['title'],'schizophrenia.md')
        self.assertGreater(trace['scored_pairs'],0)
    def test_duplicate_delete(self):
        text=b'Cardiac surgery and TAVR research describe outcomes and procedures. '*3
        doc=self.core.ingest('heart.txt',text)
        with self.assertRaisesRegex(ValueError,'已经导入'):self.core.ingest('renamed.txt',text)
        self.assertTrue(self.core.delete(doc['id']))
        self.assertEqual(self.core.stats()['chunks'],0)
        self.assertFalse(self.core.delete(doc['id']))
    def test_lexical(self):
        self.core.ingest('heart.md',b'Surgery and TAVR cardiac surgery literature with outcomes. '*3)
        with store.connect(self.core.db_path) as db:
            found=store.lexical(db,'cardiac surgery')
        self.assertTrue(found)
    def test_persistence(self):
        self.core.ingest('paper.txt',b'Known academic clinical evidence for clinical outcomes and cohort design. '*2)
        newer=ChatCore(self.core.db_path,FakeGPU())
        self.assertEqual(len(newer.list_documents()),1)
        self.assertTrue(newer.retrieve('clinical outcomes','bm25')[0])
    def test_no_documents(self):
        reply=self.core.chat('What is aortic surgery?',use_llm=False)
        self.assertFalse(reply['sources']);self.assertFalse(reply['generated'])
    def test_md_text(self):
        entries=parser.extract('note.md',('## Study\nSome research on psychosis and adverse events. '*3).encode())
        self.assertTrue(entries)
    def test_docx(self):
        payload=b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Mitral surgery report with enough research details regarding clinical patient outcomes.</w:t></w:r></w:p></w:body></w:document>'
        f=io.BytesIO()
        with zipfile.ZipFile(f,'w') as z:z.writestr('word/document.xml',payload)
        self.assertIn('Mitral',parser.extract('paper.docx',f.getvalue())[0][1])
    def test_fts_does_not_delete_other_documents(self):
        a=self.core.ingest('a.md',b'Cardiac surgery outcomes and patients. '*3)
        b=self.core.ingest('b.md',b'Mitral valve surgery outcomes and patients. '*3)
        self.core.delete(a['id'])
        src,_=self.core.retrieve('mitral valve','bm25')
        self.assertEqual(src[0]['document_id'],b['id'])

if __name__=='__main__':unittest.main()
