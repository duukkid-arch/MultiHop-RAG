"""Adapter for the existing v12.4 CUDA /api/chat; no model restarts."""
from __future__ import annotations
import json
import re
import time
import urllib.request
from typing import Sequence
from .contracts import SearchResult,Source

class LocalChatRetriever:
    def __init__(self, url='http://127.0.0.1:7866', mode='hybrid_ce', timeout=120):
        base=url.rstrip('/')
        if not re.fullmatch(r'http://(?:127\.0\.0\.1|localhost):\d+',base):
            raise ValueError('By default the adapter only calls loopback API')
        if mode not in ('bm25','hybrid','hybrid_ce'):
            raise ValueError('Invalid retrieval mode')
        self.url,self.mode,self.timeout=base,mode,timeout
    def search(self,question:str,*,agentic:bool=False,document_ids:Sequence[int]|None=None)->SearchResult:
        payload={'question':question,'mode':self.mode,'agentic':bool(agentic),'use_llm':True}
        if document_ids is not None:
            ids=list(document_ids)
            if any(type(i) is not int or i<1 for i in ids): raise ValueError('Invalid document_ids')
            payload['document_ids']=ids
        req=urllib.request.Request(self.url+'/api/chat',
             data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
             headers={'Content-Type':'application/json','Accept':'application/json'},method='POST')
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=self.timeout) as res:
            data=json.load(res)
        if not isinstance(data,dict) or 'error' in data or not isinstance(data.get('sources'),list):
            raise RuntimeError('Chat API invalid response')
        if agentic and (data.get('trace') or {}).get('agentic') is not True:
            raise RuntimeError('Server did not acknowledge agentic request')
        sources=[Source.from_api(s) for s in data['sources']]
        if document_ids is not None and not {s.document_id for s in sources}.issubset(set(document_ids)):
            raise RuntimeError('Server did not honor document_ids scope')
        return SearchResult(sources,str(data.get('answer','')),
                float(data.get('total_ms') or (time.monotonic()-start)*1000),
                data.get('trace') or {})
