# MultiHop-RAG｜面向医学论文的 Agentic RAG 系统

**Medical Literature Agentic RAG · Qwen2.5-7B · BGE-M3 · Hybrid Retrieval · Cross-Encoder · RTX 4090 D**

用于医学科研文献定位、多论文证据检索和带引用的中文问答。该项目不是医疗诊断或临床决策系统；PDF 内容、引文、数值及结论需要人工核对。

> 发布准备说明：本仓库由原 MTT S4000 / MUSA 普通 RAG 原型迭代为 v12.4 Agentic RAG。代码支持 CUDA 的前提是**已从实际 AutoDL v12.4 项目同步 `mhrag_gpu_chat_musa/gpu.py` 和 `mhrag_local_llm_musa/server.py`**。旧 GitHub 快照中的这两个文件仅支持 MUSA，不能直接用于 4090。完成文件同步后，先运行测试再发布。

## 功能

- **多跳 Agent 决策：** Qwen 根据实际检索到的原文选择 BM25、Hybrid 或 Hybrid+CrossEncoder，按证据缺口补检，并受步数和证据预算约束。
- **论文归属：** 维护研究实体与原论文映射；论文仅提到另一研究不等于已召回其原文；显式选文白名单优先。
- **容错及可观察性：** JSON 决策解析修复、规划失败兜底、重复查询控制、搜索 trace、停止原因。
- **检索及回答：** PDF/DOCX/TXT/MD 导入；SQLite FTS5、BGE-M3 嵌入、CrossEncoder 重排序、本地 Qwen2.5-7B 兼容 Chat Completions 服务；引用编号、文件和页码。
- **评测：** 同题 RAG/Agentic/Agentic Hybrid 对照；保存检索覆盖、延迟、无增益搜索、停止原因及 GPU 记录，支持断点续跑。

## v12.4 已知验收记录（示例）

RTX 4090 D 上已报告 **45/45 项离线/模拟集成测试通过**。116 篇/14,629 文本块的本地语料通过 SQLite 完整性检查；2026-10-31 的一条 Focus Pro/BESS 真机问题返回 HTTP 200，召回文献 ID 44 和 55，耗时约 19.87 s，停止原因 `evidence_budget`。**这不是 v12.4 的总体准确率，也不能证明医学结论真实可靠。**

历史 v12.1（升级前的评估工具，尚未升级线上 Agent）91 道原题中，普通 RAG 文献覆盖 92.31%，Agentic+CE 91.21%，Agentic Hybrid 86.81%；这只是目标文献召回，不是医学事实准确率。v12.4 的完整同题评测尚待执行。

## 目录结构

```text
mhrag_gpu_chat_musa/     核心检索、Agent 决策、HTTP 接口、Web UI、测试
mhrag_local_llm_musa/   本地 Qwen Chat Completions 服务
benchmarks/             12 次预备实验与 393 次有限预算对照实验
models/                 （可选公开）Qwen/BGE/CE 权重；使用 Git LFS
mhrag_gpu_chat_musa/data/knowledge.sqlite   （可选公开）文献知识库快照
results/                （可选公开）原始实验数据和报告
```

## 4090 环境设置（服务器而非 Windows）

```bash
source /root/autodl-tmp/agentic4090_env/bin/activate
export CHAT_DEVICE=cuda
export BGE_MODEL=/root/autodl-tmp/models/bge-m3
export CE_MODEL=/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
export LLM_BASE_URL=http://127.0.0.1:8002/v1
export LLM_MODEL=qwen-local
export LLM_API_KEY=local-placeholder
export CHAT_DATA_DB="$PWD/mhrag_gpu_chat_musa/data/knowledge.sqlite"
export CHAT_BATCH_SIZE=1 CHAT_CE_POOL=16 CHAT_MAX_LENGTH=384
export CHAT_AGENT_MAX_STEPS=3 CHAT_AGENT_MAX_SOURCES=4
python -u mhrag_gpu_chat_musa/app.py --host 127.0.0.1 --port 7866
```

本地 Qwen 服务应单独启动，绑定 `127.0.0.1:8002`。GPU 显存紧张时不要同时运行两套编码器/重排序器。

```bash
python -m unittest discover -s mhrag_gpu_chat_musa/tests -v
curl -sS http://127.0.0.1:7866/api/stats
```

## 评测

```bash
python benchmarks/night_v124.py --dry-run --pilot
bash benchmarks/start_pilot_v124.sh
# 预备实验健康检查后，才执行完整实验
bash benchmarks/start_night_v124.sh 9 results/v124_night
```

评估集包括 91 道原题，每模式各一次；另外有 20 道改写题和 20 道目录定位题，共 393 次调用。改写题不是独立样本；Gold 论文 ID 不传给 HTTP 服务。运行时应核对题集、服务所使用知识库以及固定分组配置。

## 数据、模型及知识产权

本项目的 `.gitignore` **不会自动排除**模型权重、知识库或实验记录；你可以将它们放在对应位置并用 Git LFS 管理。但是当前源码发布包**不包含**这些大型资料。模型分片若超过你的 Git LFS 单文件上限，需要先拆分并提供恢复清单。公开前应核对模型许可、PDF/论文全文再分发权利、日志里的个人或秘密信息及 Git LFS 费用。

此项目仅用于科研辅助、文献查找和算法实验，**不具备经过验证的临床决策能力**。不应将服务直接绑定公网，也不应存储未经授权的个人健康资料。
