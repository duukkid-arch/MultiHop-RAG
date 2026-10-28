"""Apply fail-closed v0.2 patch. Run in /root/autodl-tmp/MultiHop-RAG.
Makes timestamped backups; does not touch DB or weight files.
"""
from pathlib import Path
from datetime import datetime
import shutil

root=Path.cwd()
pkg=root/'mhrag_gpu_chat_musa'
patch=Path(__file__).resolve().parent
core=pkg/'core.py'; llm=pkg/'llm.py'
assert core.is_file() and llm.is_file(), '请在 MultiHop-RAG 根目录运行'
a=core.read_text(); b=llm.read_text()
if 'evidence_guard import' in a:
    print('补丁似乎已安装；未做任何修改。');raise SystemExit(0)
repls_a=[
("from . import store,parser,llm", "from . import store,parser,llm\nfrom .evidence_guard import topic_relevant, citation_problem"),
("if cid not in data:continue\n                    item=data[cid];did=int(item['document_id'])", 
 "if cid not in data:continue\n                    item=data[cid];did=int(item['document_id'])\n                    if not topic_relevant(question,item['filename'],item['body']):continue"),
("sources,trace=self.retrieve(question,mode,queries[1:],doc_ids)",
 "sources,trace=self.retrieve(question,mode,queries[1:],doc_ids,limit=int(os.getenv('CHAT_ANSWER_SOURCES','4')))"),
("text=llm.answer(question,sources,history=history)\n            generated=True",
 "text=llm.answer(question,sources,history=history)\n            issue=citation_problem(text,len(sources))\n            if issue:\n                text='本地小模型生成了不可核查的回答（'+issue+'）。本次已屏蔽该回答，请查看下方原文证据或换用更具体的问题。'\n                generated=False\n            else:\n                generated=True"),
("trace={'candidate_count':len(candidate),'scored_pairs':len(scored),'queries':queries,",
 "trace={'candidate_count':len(candidate),'scored_pairs':len(scored),'queries':queries,'filtered_out':len(ranked)-len(result),"),
]
repls_b=[
("'请先给出简明结论，然后说明研究证据、样本/方法/局限；相反结论也应说明。'",
 "'只给简短结论和能由当前原文直接支持的事实；无证据时只说证据不足。'"),
("'每条具体结论使用 [1] [2] 引用，引用编号只允许引用实际提供的编号。'",
 "'用纯文本回答，不要Markdown符号。每条实质性医学事实后标一个真实证据编号，例如[1]；不能连续罗列编号。'"),
("'若证据不足，请明确指出缺失信息，不要补造研究结论、统计结果或文献。'",
 "'不得用临床常识填充本次文献中缺失的信息；文献仅讨论治疗不能推断病因。'"),
("s[\"snippet\"][:1400]", "s[\"snippet\"][:650]"),
("for h in (history or [])[-6:]:", "for h in (history or [])[-2:]:"),
("return call(messages)\n", "return call(messages,max_tokens=int(os.getenv('CHAT_ANSWER_MAX_TOKENS','420')))\n"),
]
for old,new in repls_a:
    if a.count(old)!=1:raise SystemExit('原 core.py 与预期版本不同，请勿强行覆盖；未修改文件。\n未匹配: '+old[:100])
    a=a.replace(old,new,1)
for old,new in repls_b:
    if b.count(old)!=1:raise SystemExit('原 llm.py 与预期版本不同，请勿强行覆盖；未修改文件。\n未匹配: '+old[:100])
    b=b.replace(old,new,1)
compile(a,str(core),'exec');compile(b,str(llm),'exec')
compile((patch/'evidence_guard.py').read_text(),str(patch/'evidence_guard.py'),'exec')
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
for p in (core,llm):
    back=p.with_name(p.name+'.before_guard_'+stamp)
    shutil.copy2(p,back);print('备份:',back)
core.write_text(a);llm.write_text(b)
shutil.copy2(patch/'evidence_guard.py',pkg/'evidence_guard.py')
print('补丁安装成功：内容过滤 + 引用编号检查 + 缩短 Qwen 上下文。')
