import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from mhrag_local_llm_musa.server import make_handler

class StubModel:
    model_name='qwen-local'
    def generate(self,messages,max_new_tokens):
        return ('测试回答',5,3)

class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(StubModel()))
        cls.port=cls.server.server_address[1]
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def get(self,path):
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}{path}') as r:return json.load(r)
    def post(self,data):
        request=urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request) as r:return json.load(r)
    def test_health(self):
        self.assertTrue(self.get('/health')['ok'])
    def test_models(self):
        self.assertEqual(self.get('/v1/models')['data'][0]['id'],'qwen-local')
    def test_generate(self):
        ans=self.post({'model':'qwen-local','messages':[{'role':'user','content':'你好'}]})
        self.assertEqual(ans['choices'][0]['message']['content'],'测试回答')
        self.assertEqual(ans['usage']['total_tokens'],8)
    def test_refuse_wrong_model(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({'model':'not-local','messages':[{'role':'user','content':'你好'}]})
        self.assertEqual(cm.exception.code,400)
    def test_refuse_stream(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post({'model':'qwen-local','messages':[{'role':'user','content':'你好'}], 'stream':True})
        self.assertEqual(cm.exception.code,400)

if __name__=='__main__': unittest.main()
