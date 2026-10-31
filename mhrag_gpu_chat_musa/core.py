"""CUDA/MUSA interactive document Agentic RAG; independent of v6/v7 outputs."""
from __future__ import annotations
import hashlib,os,sqlite3,threading,time
from pathlib import Path
import numpy as np
from . import store,parser,llm
from .evidence_guard import topic_relevant, citation_problem
from .safe_evidence_fallback_v05 import extractive_fallback
from .answer_guard_v03 import evidence_problem
from .answer_guard_v04 import validate_and_repair
from .gpu import MUSAEncoder
from .agent import AgenticRetriever, named_comparison_terms

def named_study_terms(question):
    """Named studies found in the question; hints, NOT validated document IDs."""
    return named_comparison_terms(question)


def _exact_name(text, name):
    """Whole-word-ish casefold match for acronyms and multiword study names."""
    import re
    text = re.sub(r'[_\W]+', ' ', str(text), flags=re.UNICODE).casefold()
    name = re.sub(r'[_\W]+', ' ', str(name), flags=re.UNICODE).casefold().strip()
    return bool(name and re.search(r'(?<![a-z0-9])'+re.escape(name)+r'(?![a-z0-9])', text))


class ChatCore:
    def __init__(self,db_path=None,encoder=None):
        self.db_path=Path(db_path or os.getenv('CHAT_DATA_DB',str(Path(__file__).resolve().parent/'data/knowledge.sqlite'))).resolve()
        self.lock=threading.RLock()
        self.encoder=encoder or MUSAEncoder()
        with store.connect(self.db_path) as db:store.init(db)

    def list_documents(self):
        with store.connect(self.db_path) as db:
            rows=db.execute('SELECT d.id,d.filename,d.kind,d.created_at,COUNT(c.id) AS chunks '
                'FROM documents d LEFT JOIN chunks c ON d.id=c.document_id '
                'GROUP BY d.id ORDER BY d.id DESC').fetchall()
            return [dict(r) for r in rows]

    def stats(self):
        with store.connect(self.db_path) as db:
            return {'documents':db.execute('SELECT COUNT(*) FROM documents').fetchone()[0],
                    'chunks':db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0],
                    'gpu_loaded':self.encoder.embedder is not None if hasattr(self.encoder,'embedder') else None,
                    'reranker_loaded':self.encoder.reranker is not None if hasattr(self.encoder,'reranker') else None,
                    'llm_configured':llm.configured(),
                    'device':getattr(self.encoder,'device','musa'), 'db':str(self.db_path)}

    def ingest(self,filename,content):
        name=Path(str(filename).replace('\\','/')).name[:180].strip()
        if not name or name in ('.','..'):raise ValueError('文件名不合法')
        entries=parser.extract(name,content)
        digest=hashlib.sha256(content).hexdigest()
        with self.lock:
            with store.connect(self.db_path) as db:
                if db.execute('SELECT 1 FROM documents WHERE sha256=?',(digest,)).fetchone():
                    raise ValueError('同一份文件已经导入，避免重复索引。')
            vectors=self.encoder.embed([t for _,t in entries])
            with store.connect(self.db_path) as db:
                try:
                    did=store.add(db,name,digest,Path(name).suffix.lower().lstrip('.'),entries,vectors)
                    db.commit()
                except Exception:
                    db.rollback();raise
        return {'id':did,'filename':name,'chunks':len(entries),'pages':len(set(p for p,_ in entries if p is not None))}

    def delete(self,did):
        with self.lock:
            with store.connect(self.db_path) as db:
                n=store.remove(db,int(did));db.commit()
                return n>0

    @staticmethod
    def rrf(*rankings,k=60):
        scores={}
        for rank in rankings:
            for i,cid in enumerate(rank,1):scores[cid]=scores.get(cid,0)+1/(k+i)
        return sorted(scores,key=lambda x:(-scores[x],x))

    def retrieve(self,question,mode='hybrid_ce',queries=None,doc_ids=None,limit=8):
        if mode not in ('bm25','hybrid','hybrid_ce'):raise ValueError('检索模式不支持')
        if not question.strip():raise ValueError('问题不能为空')
        queries=list(dict.fromkeys([question]+list(queries or [])))[:4]
        ts=time.perf_counter()
        with self.lock:
            with store.connect(self.db_path) as db:
                count=db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
                if count==0:return [],{'candidate_count':0,'reason':'知识库为空','retrieval_ms':0}
                allowed=None
                if doc_ids is not None:
                    allowed={int(x) for x in doc_ids}
                    known={int(r[0]) for r in db.execute('SELECT id FROM documents')}
                    allowed &= known
                    if not allowed:return [],{'candidate_count':0,'reason':'未选择有效文档','retrieval_ms':0}
                lexical_lists=[store.lexical(db,q,100,sorted(allowed) if allowed is not None else None) for q in queries]
                dense_lists=[];max_sim={}
                if mode!='bm25':
                    vectors=db.execute('SELECT id,document_id,vector FROM chunks ORDER BY id').fetchall()
                    if allowed is not None:vectors=[r for r in vectors if int(r['document_id']) in allowed]
                    if vectors:
                        ids=[int(r['id']) for r in vectors]
                        mat=np.vstack([np.frombuffer(r['vector'],dtype=np.float32) for r in vectors])
                        query_vectors=self.encoder.embed(queries)
                        sims=query_vectors@mat.T
                        for row in sims:
                            n=min(100,len(row))
                            indices=np.argpartition(row,-n)[-n:] if n<len(row) else np.arange(len(row))
                            ordered=sorted(indices,key=lambda j:(-float(row[j]),int(ids[j])))
                            dense_lists.append([ids[j] for j in ordered])
                            for j in ordered:max_sim[ids[j]]=max(max_sim.get(ids[j],-1),float(row[j]))
                ranks=lexical_lists+dense_lists
                candidate=self.rrf(*ranks)
                if mode=='hybrid_ce':
                    pool=candidate[:int(os.getenv('CHAT_CE_POOL','40'))]
                else:pool=candidate[:80]
                if not pool:return [],{'candidate_count':0,'reason':'没有检索命中','retrieval_ms':round((time.perf_counter()-ts)*1000,1)}
                data={}
                for block in range(0,len(pool),250):
                    sub=pool[block:block+250]
                    for r in db.execute('SELECT c.id,c.document_id,c.page,c.body,d.filename FROM chunks c '
                            'JOIN documents d ON d.id=c.document_id WHERE c.id IN ('+','.join('?'*len(sub))+')',sub):
                        data[int(r['id'])]=dict(r)
                scored={}
                if mode=='hybrid_ce':
                    pairs=[(question,data[c]['body']) for c in pool if c in data]
                    scored=dict(zip([c for c in pool if c in data],self.encoder.score(pairs)))
                    ranked=sorted(scored,key=lambda c:(-scored[c],pool.index(c)))
                else:ranked=pool
                result=[];per_doc={}
                for cid in ranked:
                    if cid not in data:continue
                    item=data[cid];did=int(item['document_id'])
                    if not topic_relevant(question,item['filename'],item['body']):continue
                    if per_doc.get(did,0)>=2:continue
                    per_doc[did]=per_doc.get(did,0)+1
                    result.append({'chunk_id':cid,'document_id':did,'title':item['filename'],
                        'page':item['page'],'snippet':item['body'][:1600],
                        'semantic_similarity':round(max_sim[cid],4) if cid in max_sim else None,
                        'rerank_score':round(scored[cid],5) if cid in scored else None,
                        'citation':len(result)+1})
                    if len(result)>=limit:break
        trace={'candidate_count':len(candidate),'scored_pairs':len(scored),'queries':queries,'filtered_out':len(ranked)-len(result),
               'retrieval_ms':round((time.perf_counter()-ts)*1000,1)}
        return result,trace

    def agent_tool_search(self, question, mode='hybrid_ce', doc_ids=None,
                          limit=8, original_question=None):
        """Read-only evidence search anchored to the ORIGINAL user question.

        A named-paper scope is applied only when every recognized study can
        be mapped to a small, unambiguous set of indexed paper headers.
        Explicit doc_ids are NEVER expanded.  Unknown names fall back to the
        normal retrieval procedure rather than silently dropping evidence.
        """
        original = str(original_question or question)
        names = named_study_terms(original)
        allowed = {int(x) for x in doc_ids} if doc_ids is not None else None
        entity_scope = None
        resolved = {}
        if names:
            with self.lock:
                with store.connect(self.db_path) as db:
                    docs = list(db.execute('SELECT id,filename FROM documents'))
                    # PDF first pages contain article titles / abstracts.  Do
                    # not mine References at the end: that confuses cited
                    # trials with the article itself.
                    heads = {}
                    for r in db.execute('SELECT document_id,body FROM chunks '
                                        'WHERE ordinal < 3 ORDER BY document_id,ordinal'):
                        did = int(r['document_id'])
                        heads[did] = (heads.get(did,'') + ' ' + str(r['body'])[:1100])[:3300]
                    for name in names:
                        ids = {int(d['id']) for d in docs
                               if _exact_name(d['filename'],name) or
                                  _exact_name(heads.get(int(d['id']),''),name)}
                        # A term occurring in >6 papers is a topic, not a
                        # reliably unique paper identity (e.g. NeuroAI).
                        if 0 < len(ids) <= 6:
                            resolved[name] = ids
            if len(resolved) == len(names):
                # A follow-up naming exactly one study should search only that
                # study, not the other comparison arm.  This prevents the CE
                # reranker from repeatedly spending the citation budget on A.
                named_in_query = [term for term in names if _exact_name(question,term)]
                if len(named_in_query) == 1:
                    scope = resolved[named_in_query[0]]
                else:
                    scope = set().union(*resolved.values())
                if len(scope) <= 8:
                    entity_scope = sorted(scope if allowed is None else scope & allowed)
        if entity_scope is not None and not entity_scope:
            return [], {'candidate_count':0,'retrieval_ms':0,'scored_pairs':0,
                        'entity_scope':[],'entity_names':names,
                        'entity_document_map':{k:sorted(v) for k,v in resolved.items()},
                        'original_query_fused':False,
                        'reason':'用户选择的文献与研究名称限定范围不相交'}
        effective_scope = entity_scope if entity_scope is not None else doc_ids
        # Combine the user's complete intent with the planner's targeted
        # follow-up: retrieval uses both; CE ranks by the actual user task.
        if original != question:
            found, trace = self.retrieve(original,mode,queries=[question],
                                          doc_ids=effective_scope,limit=limit)
        else:
            found, trace = self.retrieve(question,mode,doc_ids=effective_scope,limit=limit)
        trace = dict(trace)
        trace.update({'entity_scope':entity_scope,
                      'entity_names':names if entity_scope is not None else [],
                      'entity_document_map':{k:sorted(v) for k,v in resolved.items()},
                      'original_query_fused':original != question})
        return found, trace

    def chat(self,question,mode='hybrid_ce',use_llm=True,plan=False,doc_ids=None,history=None,agentic=False):
        if not isinstance(question,str) or not 1<=len(question.strip())<=2500:
            raise ValueError('问题长度应为 1–2500 个字符')
        start=time.perf_counter()
        if agentic:
            if not use_llm:raise ValueError('Agentic RAG 需要开启大模型回答')
            sources,trace=AgenticRetriever(self).run(question,mode,doc_ids,history)
            planned=len(trace['queries'])>1
        else:
            queries=llm.plan(question) if plan and use_llm and llm.configured() else [question]
            sources,trace=self.retrieve(question,mode,queries[1:],doc_ids,limit=int(os.getenv('CHAT_ANSWER_SOURCES','4')))
            planned=len(queries)>1
            trace['agentic']=False
        retrieval_ms=trace['retrieval_ms']
        if not sources:
            text='知识库中未找到可核查的证据。请确认 PDF 文字提取成功、选择了正确的论文，或更换检索问题。'
            generated=False
        elif use_llm and llm.configured():
            text=llm.answer(question,sources,history=history)
            text,issue=validate_and_repair(question,sources,text,llm.answer,citation_problem,evidence_problem)
            if issue:
                text=extractive_fallback(sources,issue)
                generated=False
            else:
                generated=True
        else:
            text='尚未启用大模型回答。已找到以下证据，可查看来源论文和页码。'
            generated=False
        return {'answer':text,'sources':sources,'mode':mode,'generated':generated,'agentic':agentic,'trace':trace,
                'retrieval_ms':retrieval_ms,'total_ms':round((time.perf_counter()-start)*1000,1),
                'llm_enabled':llm.configured(),'planned':planned}
