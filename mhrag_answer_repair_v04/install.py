"""Install v0.4 citation repair on top of v0.3; no DB/weights modified."""
from pathlib import Path
from datetime import datetime
import shutil
import tempfile
import os

root = Path.cwd()
pkg = root / 'mhrag_gpu_chat_musa'
core = pkg / 'core.py'
addon = Path(__file__).with_name('answer_guard_v04.py')
if not core.is_file():
    raise SystemExit('请在 MultiHop-RAG 项目根目录执行；未修改任何文件。')
original = core.read_text(encoding='utf-8')
if 'from .answer_guard_v04 import validate_and_repair' in original:
    print('v0.4 已经安装，未重复修改。')
    raise SystemExit(0)
old_import = 'from .answer_guard_v03 import evidence_problem'
old_check = 'issue=citation_problem(text,len(sources)) or evidence_problem(question,text,sources)'
if original.count(old_import) != 1 or original.count(old_check) != 1:
    raise SystemExit('没有找到预期的 v0.3 代码，停止安装。请先检查 v0.3 是否完成安装。')
new = original.replace(old_import,
    old_import + '\nfrom .answer_guard_v04 import validate_and_repair', 1)
new = new.replace(old_check,
    'text,issue=validate_and_repair(question,sources,text,llm.answer,citation_problem,evidence_problem)', 1)
compile(new, str(core), 'exec')
compile(addon.read_text(encoding='utf-8'), str(addon), 'exec')
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup = core.with_name(f'core.py.before_answer_repair_v04_{stamp}')
shutil.copy2(core, backup)
# Write helper first, then atomically replace core so imports are always available.
shutil.copy2(addon, pkg / 'answer_guard_v04.py')
fd, temp_name = tempfile.mkstemp(prefix='.core_v04_', dir=str(pkg))
try:
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(new)
    os.replace(temp_name, core)
finally:
    if os.path.exists(temp_name): os.unlink(temp_name)
print('安装完成。仅修改了:', core, '及 answer_guard_v04.py')
print('备份文件:', backup)
print('只需重启端口 7865；不要关闭端口 8001 或清理知识库。')
