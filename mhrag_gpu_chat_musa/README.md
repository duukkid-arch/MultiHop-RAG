# MultiHop-RAG · MUSA GPU 医学文献智能问答（体验版）

> **独立于 v6/v7 科研实验**：使用新的 `mhrag_gpu_chat_musa/data/knowledge.sqlite`，不会修改既有 `results/`、`dataset/`、`models/`。这是**单用户本地原型**，在实际 MTT S4000 上完成烟雾测试前，不应声称已完成生产环境验收。

## 当前功能

- **PDF、DOCX、TXT、MD** 导入，文本型 PDF 保留页码；一次可选多文件（单文件不超过 20MB / 150 页）。扫描图片型 PDF 不支持 OCR，复杂双栏和图表仍需人工检查。
- 保存 SQLite FTS5 文本与 BGE-M3 1024 维归一化语义向量，重复文件拦截，支持删除、重启持久化。
- BM25 基线 / BGE-M3 + BM25 混合检索 / MUSA Cross-Encoder @40 三种模式。
- 中文提问检索英文医学论文（依赖 BGE-M3 的多语言模型能力），结果展示**文件、PDF 页码和原文片段**。
- 可选 **OpenAI-compatible Chat Completions** LLM（OpenAI、DeepSeek、通义千问兼容服务等，具体需看厂商接口）；可选 LLM 问题分解（最多 3 个子问题，非已验证的多跳算法），并生成带 `[1]` 引用的中文答案。
- 不配置 LLM 时仍能使用 GPU 检索和证据展示。
- 提供离线 CPU 测试（mock encoder）与 MUSA GPU 实际 smoke 检查。

**边界：** 本项目不是临床决策工具，不保证引用无误，不是 v6/v7 的严格质量对照；缺少完整 RAG 评估/内容治理/用户登录/抗滥用/扫描 OCR/流式首字延迟；**请勿直接对公网开放**。

## 0. 先暂停 v6，释放显存并保留检查点

在当前 MTT S4000 `autodl-container-5c854faafe-34d1c50b`，注意先查看状态；之前 v6 的主进程是 PID `79544`，现在**不保证还在运行**：

```bash
cd /root/autodl-tmp/MultiHop-RAG
cat results/v6_focus_750_musa/STATUS
ps -p 79544 -o pid,etime,stat,cmd
python mhrag_gpu_chat_musa/tools/pause_v6.py --pid 79544
```

如果状态是 `DONE`，完全不用停止。如果 `RUNNING` 且脚本确认 PID 正确，再执行：

```bash
python mhrag_gpu_chat_musa/tools/pause_v6.py --pid 79544 --stop
python mhrag_gpu_chat_musa/tools/verify_checkpoint.py
ps -p 79544 -o pid,cmd
```

脚本先暂时冻结调度器，终止**该调度器的子任务**，之后让调度器自行记录 `FAILED` 并退出。这是**可恢复的受控中断**：`FAILED` 在此情形不代表数据丢失。它有意不使用 `kill -9`；也不依赖 `SIGSTOP` 长期挂起（后者仍占用 VRAM）。正在计算的问题可能需要重做，但之前已落盘的有效 JSONL 行会复用。若检查脚本提示 `BAD` 尾行，先不要恢复；阅读下方修复说明。

在启用 chat 前确认不再有对应 GPU 推理任务，避免争抢 GPU：

```bash
ps -ef | grep -E '[m]hrag_v6_musa|[r]erank_musa.py'
```

## 1. 环境前提

这是 **MUSA 版**，不是 CUDA 版。之前已经通过：

```text
MUSA: MTT S4000
BGE success (2, 1024)
CE success ...
```

上述是**原 v6 脚本**的 smoke 测试，新的聊天集成需要单独再测。

进入 **新 MTT S4000 实例**：

```bash
cd /root/autodl-tmp/MultiHop-RAG
python -c "import torch, torch_musa, numpy, sentence_transformers; print(torch.musa.is_available())"
python -m mhrag_gpu_chat_musa.app --check
```

应看到 `MUSA: MTT S4000`、`BGE: (2,1024)`、`CE: [score]`。首次模型加载有启动开销。

PDF 有 3 种解析后备方案：PyMuPDF、pypdf、系统 `pdftotext`。先检查：

```bash
python -c "import fitz; print('PyMuPDF OK')" || command -v pdftotext || python -c "import pypdf; print('pypdf OK')"
```

若三种都没有，**只安装文档解析库**：

```bash
python -m pip install --no-cache-dir 'PyMuPDF>=1.24,<2'
```

**绝对不要在当前 MUSA 环境中无差别执行 `pip install -r requirements...` 并升级 torch。**

## 2. 可选：接入 OpenAI 或其他 LLM API

这套程序使用通用 **Chat Completions** 接口，而非 File Search / Responses API：先本地检索，只把最多 8 段必要证据发送给远程服务。

在**新 MTT 实例**的终端安全读取密钥：

```bash
read -rsp '输入 LLM API Key: ' LLM_API_KEY; echo
export LLM_API_KEY
export LLM_BASE_URL='https://api.openai.com/v1'
export LLM_MODEL='gpt-4.1-mini'    # 可根据你的 API 模型权限更换
```

使用其他提供商时改 `LLM_BASE_URL`、`LLM_MODEL`；服务需要支持 `POST /chat/completions` 和对应消息协议。不要在命令历史、GitHub 或聊天里公开密钥；如果未设置 key，网页会提示「LLM 未配置」，仍能进行 GPU 检索。

**隐私：** 生成答案时会把检索到的原文片段发给云端。只上传有权处理的文献，不要导入包含真实患者个人信息的病历。API 调用可能产生费用。网页关闭、重新开新的 shell、或 screen 重新启动后，要重新设置环境变量。

## 3. 启动网页

```bash
cd /root/autodl-tmp/MultiHop-RAG
screen -S gpu_chat_musa
# 进入 screen 后，在同一 shell 中 export 密钥和模型环境变量
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BGE_MODEL=/root/autodl-tmp/models/bge-m3
export CE_MODEL=/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
python -u mhrag_gpu_chat_musa/app.py --host 127.0.0.1 --port 7865
```

如果你在 `screen` 外设置了 LLM key，而进入新的 screen 后丢失，请在 screen 内再次 `read -rsp ...` 和 `export`。使用 `Ctrl+A` 然后 `D` 从 screen 脱离。

健康检查：

```bash
curl -s http://127.0.0.1:7865/api/health
```

## 4. 用自己的电脑访问

新 MTT S4000 的 SSH（以前的已验证入口）：

```bash
ssh -N -L 7865:127.0.0.1:7865 -p 21207 root@connect.sha1.seetacloud.com
```

在**自己电脑**浏览器访问 http://127.0.0.1:7865 。确保服务只绑定 `127.0.0.1`。不要和**旧无卡 4090D** 的 `11719` 端口混淆。

网页：左侧上传医学 PDF；勾选参与检索的论文；选择 Hybrid + CE；勾选「大模型综合回答」。如果云端 API 具备访问权限，会基于最多 8 条候选证据回答。可以开启「LLM 分解检索问题」，这会多消耗一次 API 调用。

推荐起步问题：

- 根据已上传论文，氯氮平对难治性精神分裂症的主要疗效证据有哪些？请标明论文和页码。
- 与微创二尖瓣外科相比，M-TEER 的研究报告了哪些结局及局限？
- 当前论文是否足以比较不同术式的长期死亡率？没有直接可比的数据就明确说明。

**实验应选择同一批真实 PDF，人工核对页码和原文，再评价回答正确性。**

## 5. 在独立环境运行自动化测试

```bash
cd /root/autodl-tmp/MultiHop-RAG
python -m unittest discover -s mhrag_gpu_chat_musa/tests -p 'test*.py' -v
```

不需要 GPU 的测试使用 Mock Encoder，不验证真实 MUSA kernel 或在线 LLM。PDF 测试仅在安装 PyMuPDF 时运行。

## 6. 恢复 v6

停止聊天服务（在 screen 内 `Ctrl+C`），确认 GPU 内存已释放，然后：

```bash
cd /root/autodl-tmp/MultiHop-RAG
python mhrag_gpu_chat_musa/tools/verify_checkpoint.py
# 若只有最后一行损坏（并确认是终止造成的），会备份并修复：
# python mhrag_gpu_chat_musa/tools/verify_checkpoint.py --repair-tail
bash mhrag_gpu_chat_musa/tools/resume_v6.sh
bash mhrag_gpu_chat_musa/tools/resume_v6.sh --yes
```

调度器会跳过已有 `DONE` 阶段；CE 评分脚本自动跳过检查点中已有的完整记录。**请先检查不会同时启动第二个 v6 进程。**

## 7. 发布到 GitHub 前的检查

- 只提交本目录的源码、`README`、测试、`.env.example`；**不要上传** `data/`、医学论文、结果、模型文件、API Key 或 Hugging Face 缓存。
- 测试 PDF 上传，查验是否有真实正文和页码；运行真实 MUSA `--check`；用有权限的 LLM API 生成几次回答。
- 写明支持范围和未验证点，不要将医疗助手声明为医疗诊断/临床决策工具。
- 公共仓库中不应包含未经授权的第三方医学论文全文或从私有系统导出的历史对话。
- 请选择适合自己的开源许可后再公开发布。

```bash
cd /root/autodl-tmp/MultiHop-RAG/mhrag_gpu_chat_musa
find . -maxdepth 2 -type f | sort
git init
# 自行设置仓库地址并确认隐私/许可证后，再执行 git add / commit / push。
```

## 局限与后续路线

首次导入会在 GPU 上为论文生成向量。每次检索会从 SQLite 读取该知识库的向量并用 NumPy 计算相似度，**适合小批量科研文献（如 10～50 篇）而非海量论文**；扩大规模需换成真正的向量索引。LLM 问题分解是启发式检索规划，不是已经在独立数据集验证的完整多跳算法；医学结果、引用仍需核查，缺少复杂医学表格与图像的准确理解。
