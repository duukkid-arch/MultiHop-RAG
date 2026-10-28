"""Single-user local-only HTTP API and web app, no FastAPI dependency."""
from __future__ import annotations
import argparse,json,logging,os,sys,traceback,urllib.parse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

if __package__ is None or __package__=='':
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from mhrag_gpu_chat_musa.core import ChatCore
from mhrag_gpu_chat_musa.llm import LLMError

ROOT=Path(__file__).resolve().parent
WEB=ROOT/'web'

class Handler(BaseHTTPRequestHandler):
    core=None
    server_version='MHRAG-MUSA/0.1'

    def respond(self,status,payload):
        body=json.dumps(payload,ensure_ascii=False).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        self.end_headers();self.wfile.write(body)

    def file(self,name,kind):
        b=(WEB/name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(b)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(b)

    def read_body(self,limit):
        try:n=int(self.headers.get('Content-Length','-1'))
        except ValueError:raise ValueError('无效 Content-Length')
        if n<0 or n>limit:raise ValueError('请求体大小无效或超过上限')
        return self.rfile.read(n)

    def do_GET(self):
        path=urllib.parse.urlparse(self.path).path
        try:
            if path=='/':return self.file('index.html','text/html; charset=utf-8')
            if path=='/app.js':return self.file('app.js','application/javascript; charset=utf-8')
            if path=='/style.css':return self.file('style.css','text/css; charset=utf-8')
            if path=='/api/health':return self.respond(200,{'ok':True,'backend':'musa','llm_configured':self.core.stats()['llm_configured']})
            if path=='/api/stats':return self.respond(200,self.core.stats())
            if path=='/api/documents':return self.respond(200,{'documents':self.core.list_documents()})
            return self.respond(404,{'error':'Not found'})
        except Exception as exc:
            logging.exception('GET failed');return self.respond(500,{'error':'服务器内部错误'})

    def do_POST(self):
        path=urllib.parse.urlparse(self.path).path
        try:
            if path=='/api/upload':
                rawname=self.headers.get('X-Filename','')
                filename=urllib.parse.unquote(rawname)
                content=self.read_body(20*1024*1024)
                return self.respond(201,self.core.ingest(filename,content))
            if path=='/api/chat':
                body=json.loads(self.read_body(24000))
                if not isinstance(body,dict):raise ValueError('JSON 对象格式错误')
                ids=body.get('document_ids')
                if ids is not None and (not isinstance(ids,list) or len(ids)>500 or not all(isinstance(x,int) for x in ids)):
                    raise ValueError('document_ids 必须是整数 ID 数组')
                history=body.get('history',[])
                if not isinstance(history,list) or len(history)>20:raise ValueError('对话历史格式错误')
                return self.respond(200,self.core.chat(
                    body.get('question',''),body.get('mode','hybrid_ce'),
                    body.get('use_llm',True) is True,body.get('plan',False) is True,
                    ids,history))
            return self.respond(404,{'error':'Not found'})
        except (ValueError,UnicodeError,KeyError,json.JSONDecodeError) as exc:
            return self.respond(400,{'error':str(exc)[:350]})
        except LLMError as exc:
            return self.respond(502,{'error':str(exc)[:350]})
        except Exception:
            logging.exception('POST failed')
            return self.respond(500,{'error':'操作失败，请查看服务端日志。'})

    def do_DELETE(self):
        path=urllib.parse.urlparse(self.path).path
        try:
            if not path.startswith('/api/documents/'):return self.respond(404,{'error':'Not found'})
            did=int(path.rsplit('/',1)[-1])
            if not self.core.delete(did):return self.respond(404,{'error':'文件不存在'})
            return self.respond(200,{'deleted':True,'id':did})
        except ValueError:return self.respond(400,{'error':'无效文档 ID'})
        except Exception:
            logging.exception('DELETE failed');return self.respond(500,{'error':'删除失败'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=7865)
    p.add_argument('--check',action='store_true',help='Load BGE and CE on MUSA and score small test')
    args=p.parse_args()
    core=ChatCore()
    if args.check:
        print('MUSA:',core.encoder.check())
        print('BGE:',core.encoder.embed(['氯氮平治疗精神分裂症','Clozapine in schizophrenia']).shape)
        print('CE:',core.encoder.score([('氯氮平的疗效','Clozapine may be effective in treatment resistant schizophrenia')]))
        return
    Handler.core=core
    if args.host not in ('127.0.0.1','localhost','::1'):
        print('WARNING: This prototype has no authentication; use 127.0.0.1 and SSH forwarding instead.',file=sys.stderr)
    print(f'MultiHop-RAG MUSA Chat | http://{args.host}:{args.port} | DB={core.db_path}',flush=True)
    with ThreadingHTTPServer((args.host,args.port),Handler) as server:
        server.daemon_threads=True
        try:server.serve_forever()
        except KeyboardInterrupt:pass

if __name__=='__main__':main()
