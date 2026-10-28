# MultiHop-RAG v0.4 · 引用完整性修复

用于 **已成功安装 v0.2 + v0.3 的** `mhrag_gpu_chat_musa`。解决当前实测的漏洞：短答案 `aHR=0.68...` 因不足 80 字而绕过引用检查；提高 1.5B 本地模型引用格式失败后的恢复机会。

**具体行为**
- 任何包含事实性回答、但没有 `[1]` 等引用的输出都被拒绝，不再有 80 字豁免。短小的明确拒答除外。
- 模型仅因为引用缺失或编号错误失败时，自动重写**一次**，然后再次验证。不满足要求仍显示原有的「不可核查回答已屏蔽」。可通过 `CHAT_CITATION_RETRY=0` 禁用第二次生成。
- `aHR=...` / `HR=...` / `OR=...` / `RR=...` 数值须出现在某条检索证据中，否则拦截。此规则**不验证统计结局归属或因果解释**，仍须人工校对。
- 不会自动向答案追加 `[1]`：凭空加引用会掩盖幻觉。

**安装（AutoDL MTT S4000，MultiHop-RAG 根目录）**

```bash
unzip -o mhrag_answer_repair_v04.zip
python mhrag_answer_repair_v04/test_repair.py
python mhrag_answer_repair_v04/install.py
python -m unittest discover -s mhrag_gpu_chat_musa/tests -v
```

仅对磁盘文件打补丁；**仅在准备好时停止并重启 7865 网页服务**以加载它。保留 8001 Qwen 服务、数据库、模型权重及 v6 实验。安装器自动备份 `core.py.before_answer_repair_v04_*`，重复安装会跳过。

**部署前验收**

```bash
python -m py_compile mhrag_gpu_chat_musa/core.py mhrag_gpu_chat_musa/answer_guard_v04.py
```

随后在网页测试：
1. 简单量化问题——必须有对应 `[1]` 引用，数值在证据中找到。
2. 无法依据上传证据回答的问题——应明确说证据不足。
3. 复杂问题——仍可能因本地 1.5B 模型能力有限而被屏蔽；此时不要把被屏蔽理解成检索失败。

**限制**：不是临床级事实核查。无法检测所有虚假陈述、数字错配、文献偏倚或临床风险遗漏；更强的生成模型和结构化事实核验仍很重要。
