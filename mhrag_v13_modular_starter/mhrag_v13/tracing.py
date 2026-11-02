"""JSONL trajectory logging without prompts, source text, API keys or raw user data."""
from __future__ import annotations
import json
import threading
from datetime import datetime,timezone
from pathlib import Path

class JsonlTrace:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
    def emit(self,event):
        record={'ts':datetime.now(timezone.utc).isoformat(),**event}
        with self.lock,self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')

class NullTrace:
    def emit(self,event): pass
