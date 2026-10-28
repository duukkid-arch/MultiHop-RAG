from __future__ import annotations
import json,sys,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mhrag_gpu_chat_musa.app import Handler
from mhrag_gpu_chat_musa.core import ChatCore
from test_app import FakeGPU

class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        Handler.core=ChatCore(Path(cls.tmp.name)/'server.sqlite',FakeGPU())
        cls.http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.url='http://127.0.0.1:'+str(cls.http.server_address[1])
        cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close();cls.tmp.cleanup()
    def fetch(self,path,method='GET',data=None,headers=None):
        req=urllib.request.Request(self.url+path,data=data,headers=headers or {},method=method)
        with urllib.request.urlopen(req,timeout=5) as f:return f.status,json.load(f)
    def test_health_and_ui(self):
        status,data=self.fetch('/api/health')
        self.assertEqual(status,200)
        self.assertEqual(data['backend'],'musa')
        with urllib.request.urlopen(self.url+'/') as f:self.assertIn(b'html',f.read().lower())
    def test_upload_query_delete(self):
        text=b'Clozapine treatment resistant schizophrenia and safety. '*3
        status,doc=self.fetch('/api/upload','POST',text,{'X-Filename':'test_paper.md'})
        self.assertEqual(status,201)
        self.assertGreater(doc['chunks'],0)
        status,result=self.fetch('/api/chat','POST',json.dumps({'question':'clozapine schizophrenia','use_llm':False,'mode':'hybrid_ce','document_ids':[doc['id']]}).encode(),{'Content-Type':'application/json'})
        self.assertFalse(result['generated'])
        self.assertEqual(result['sources'][0]['title'],'test_paper.md')
        self.assertEqual(self.fetch('/api/documents/'+str(doc['id']),'DELETE')[0],200)

class PDFTest(unittest.TestCase):
    def test_real_pdf(self):
        try:import fitz
        except ImportError:self.skipTest('PyMuPDF missing')
        from mhrag_gpu_chat_musa import parser
        doc=fitz.open();page=doc.new_page()
        page.insert_text((70,80),'Clozapine may be studied for treatment resistant schizophrenia and safety. ' *2)
        b=doc.tobytes();doc.close()
        entries=parser.extract('paper.pdf',b)
        self.assertTrue(entries)
        self.assertEqual(entries[0][0],1)
        self.assertIn('Clozapine',entries[0][1])

if __name__=='__main__':unittest.main()
