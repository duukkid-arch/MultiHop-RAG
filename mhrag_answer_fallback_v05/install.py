"""Upgrade v0.2–v0.4 guard to fail-safe verbatim evidence fallback."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import shutil
import tempfile

root = Path.cwd()
pkg = root / 'mhrag_gpu_chat_musa'
core = pkg / 'core.py'
addon = Path(__file__).resolve().with_name('safe_evidence_fallback_v05.py')
if not core.is_file():
    raise SystemExit('请在 MultiHop-RAG 项目根目录执行。未修改文件。')
s = core.read_text(encoding='utf-8')
if 'from .safe_evidence_fallback_v05 import extractive_fallback' in s:
    print('v0.5 已安装，无需重复安装。')
    raise SystemExit(0)
import_line = 'from .evidence_guard import topic_relevant, citation_problem'
old = "text='本地小模型生成了不可核查的回答（'+issue+'）。本次已屏蔽该回答，请查看下方原文证据或换用更具体的问题。'"
if s.count(import_line) != 1 or s.count(old) != 1:
    raise SystemExit('当前 core.py 不符合 v0.2～v0.4 预期版本。未修改文件，请勿覆盖。')
new = s.replace(import_line, import_line + '\nfrom .safe_evidence_fallback_v05 import extractive_fallback', 1)
new = new.replace(old, 'text=extractive_fallback(sources,issue)', 1)
compile(new, str(core), 'exec')
compile(addon.read_text(encoding='utf-8'), str(addon), 'exec')
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup = core.with_name(f'core.py.before_extract_fallback_v05_{stamp}')
shutil.copy2(core, backup)
shutil.copy2(addon, pkg / 'safe_evidence_fallback_v05.py')
fd, temp_name = tempfile.mkstemp(prefix='.core_v05_', dir=str(pkg))
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(new)
    os.replace(temp_name, core)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
print('v0.5 安装完成。备份：', backup)
print('原来的主题过滤、引用校验和本地 Qwen 设置均保留。')
print('仅需在合适时间重启 7865 网页服务，8001 Qwen 不动。')
