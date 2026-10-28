"""Small SQLite document/index store. Own database, isolated from v6 evaluation files."""
from __future__ import annotations
import hashlib, re, sqlite3
from contextlib import contextmanager
from pathlib import Path

WORD = re.compile(r'[A-Za-z0-9]+|[\u3400-\u9fff]+')
STOP = set('the a an of to and or in on for with as is are was were from by what how who where when why whether compare please study studies research review paper which effects among between that this'.split())

def tokens(text):
    vals=[]
    for w in WORD.findall(str(text).lower()):
        if re.fullmatch(r'[\u3400-\u9fff]+',w):
            vals.extend(w[i:i+2] for i in range(len(w)-1)) if len(w)>1 else vals.append(w)
        elif len(w)>1 and w not in STOP: vals.append(w)
    return list(dict.fromkeys(vals))

def chunk(text,size=750,overlap=150):
    text=str(text).strip()
    if not text:return []
    out=[]; left=0
    while left<len(text):
        right=min(len(text),left+size)
        if right<len(text):
            for sep in ('\n\n','\n','. ','。','; ','；'):
                pos=text.rfind(sep,max(left+size//2,right-135),right)
                if pos>=0:
                    right=pos+len(sep);break
        if right<=left: break
        part=text[left:right].strip()
        if part:out.append(part)
        if right>=len(text):break
        left=max(left+1,right-overlap)
    return out

@contextmanager
def connect(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(path),timeout=30)
    try:
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        with db:yield db
    finally:db.close()

def init(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY,filename TEXT NOT NULL,sha256 TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY,document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,page INTEGER,body TEXT NOT NULL,vector BLOB NOT NULL);
    CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(document_id);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(words,tokenize='unicode61');
    ''')
    db.commit()

def add(db,filename,digest,kind,entries,embeddings):
    """entries: list [(page:int|None, text:str)], embeddings normalized ndarray float32."""
    if len(entries)!=len(embeddings):raise ValueError('embedding/chunk count mismatch')
    cur=db.execute('INSERT INTO documents(filename,sha256,kind) VALUES(?,?,?)',(filename,digest,kind))
    did=cur.lastrowid
    import numpy as np
    for pos,((page,text),vector) in enumerate(zip(entries,embeddings)):
        buf=np.asarray(vector,dtype=np.float32).tobytes()
        c=db.execute('INSERT INTO chunks(document_id,ordinal,page,body,vector) VALUES(?,?,?,?,?)',(did,pos,page,text,buf))
        db.execute('INSERT INTO chunks_fts(rowid,words) VALUES(?,?)',(c.lastrowid,' '.join(tokens(filename+' '+text))))
    return did

def remove(db,did):
    ids=[r[0] for r in db.execute('SELECT id FROM chunks WHERE document_id=?',(did,))]
    for cid in ids:db.execute('DELETE FROM chunks_fts WHERE rowid=?',(cid,))
    n=db.execute('DELETE FROM documents WHERE id=?',(did,)).rowcount
    return n

def lexical(db,query,limit=100,doc_ids=None):
    words=tokens(query)[:24]
    if not words:return []
    query=' OR '.join('"'+w.replace('"','')+'"' for w in words)
    sql='SELECT c.id FROM chunks_fts f JOIN chunks c ON c.id=f.rowid WHERE f.words MATCH ?'
    args=[query]
    if doc_ids is not None:
        if not doc_ids:return []
        sql+=' AND c.document_id IN ('+','.join('?'*len(doc_ids))+')'
        args.extend(doc_ids)
    sql+=' ORDER BY bm25(f) LIMIT ?';args.append(limit)
    # FTS5 MATCH and bm25 require the virtual table name, not an alias.
    sql=sql.replace('chunks_fts f JOIN','chunks_fts JOIN').replace('f.rowid','chunks_fts.rowid').replace('f.words MATCH','chunks_fts.words MATCH').replace('bm25(f)','bm25(chunks_fts)')
    return [int(r[0]) for r in db.execute(sql,args)]
