#!/usr/bin/env python3
"""Read-only check of source tree before claiming CUDA-ready public release."""
from pathlib import Path
import sys, re

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd()
errors=[];warnings=[]
required=['README.md','mhrag_gpu_chat_musa/agent.py','mhrag_gpu_chat_musa/core.py',
'mhrag_gpu_chat_musa/llm.py','mhrag_gpu_chat_musa/gpu.py',
'mhrag_local_llm_musa/server.py','.gitattributes']
for f in required:
    if not (root/f).is_file():errors.append('缺少必需文件: '+f)
for f in ['README.md','mhrag_gpu_chat_musa/agent.py','mhrag_gpu_chat_musa/core.py','mhrag_gpu_chat_musa/llm.py']:
    p=root/f
    if p.exists() and re.search(r'(?m)^\s*(<<<<<<< |=======\s*$|>>>>>>> )',p.read_text(encoding='utf-8',errors='replace')):
        errors.append('Git 合并冲突标记: '+f)
p=root/'mhrag_gpu_chat_musa/gpu.py'
if p.is_file():
    code=p.read_text(encoding='utf-8',errors='replace')
    if 'torch.cuda.is_available' not in code or 'cuda' not in code:errors.append('gpu.py 仍是旧版 MUSA-only；从当前 4090 服务器同步')
p=root/'mhrag_local_llm_musa/server.py'
if p.is_file():
    code=p.read_text(encoding='utf-8',errors='replace')
    if "'--device'" not in code and '"--device"' not in code:errors.append('server.py 仍是旧版 MUSA-only；从当前 4090 服务器同步')
for file in ['mhrag_gpu_chat_musa/data/knowledge.sqlite','models/Qwen2.5-7B-Instruct']:
    if not (root/file).exists():warnings.append('尚未复制可选大文件: '+file)
print('检查项目:',root)
for s in errors:print('ERROR:',s)
for s in warnings:print('提醒:',s)
print('结果:', 'FAIL' if errors else 'PASS (源码结构/兼容性静态检查)')
sys.exit(1 if errors else 0)
