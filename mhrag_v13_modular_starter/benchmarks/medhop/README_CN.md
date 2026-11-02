# MultiHop-RAG v13 | MedHop 公开多跳基准

**状态：数据准备工具已实现，尚未在 AutoDL 网络环境下载验证，六实验臂的 MedHop GPU 适配尚未完成。**

GitHub 规范集成：`DATASET_CARD.md`、标准化输入/评分器、Hugging Face 下载镜像、离线测试和 CI。仓库**不保存原始医学 PDF、已有 SQLite 库、MedHop 原始数据及答案文件**。

## 为什么不用 340 MB Zenodo ZIP？

AutoDL 到 Zenodo 已报告 `curl: (7) Connection refused`；此工具优先访问 Hugging Face BigBio 的 `medhop_source` 转换版 Parquet（不是将所有 WikiHop+MedHop 下载回来），默认 `https://huggingface.co`，失败再试 `https://hf-mirror.com`。数据可复现，下载 SHA256 存于私有清单。运行仍需 AutoDL 能访问至少一个来源；**工具未验证你的服务器可以访问 HF**。

仅使用 `medhop_source`：它保存原始 MedHop 每题多份 `supports` 文档。`medhop_bigbio_qa` 把这些文档合并成单段 `context`，不满足多跳逐证据评价需要，所以不会自动降级使用。

## AutoDL：最少执行方式

将此文件夹复制到 AutoDL 的项目根目录对应路径：

```text
mhrag_v13_modular_starter/benchmarks/medhop/
```

执行：

```bash
cd /root/autodl-tmp/agentic_4090_v12_4/MultiHop-RAG-main
bash mhrag_v13_modular_starter/benchmarks/medhop/setup_medhop_autodl.sh
```

如果 HF 官方域名被阻断，可再试一次镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com bash mhrag_v13_modular_starter/benchmarks/medhop/setup_medhop_autodl.sh
```

该脚本只在私有 `/root/autodl-tmp/medhop_benchmark_private` 中建立独立 Python 虚拟环境安装 `pyarrow`，不修改现有 GPU 服务或 116-PDF 知识库。若已经有 `medhop_original/train.json` / `dev.json`，自动跳过下载。

完成后生成 `train_questions_NO_GOLD.jsonl`、`train_context_corpus.jsonl`、`train_gold_PRIVATE.jsonl` 及对应 dev 文件和 `manifest.json`，不会覆盖同名已存在目录。

**注意**：HF Parquet 可访问性未经过 AutoDL 实测。如果网络同时封锁 HF/HF 镜像，需要在 Windows 浏览器从官方/镜像下载相应源格式数据后传到 AutoDL。不要重复运行旧的 Zenodo-only 下载脚本。

## 本地代码测试（无网络、无 GPU）

```bash
python -m unittest discover -s mhrag_v13_modular_starter/benchmarks/medhop/tests -v
```

## 六臂实验协议

`case_id + question + candidates + corpus_scope` 是题面；每个 `case_id` 仅检索对应的 `supports` 子语料。推理进程**不得读取** `*_gold_PRIVATE.jsonl`。F、旧 A、C、E1、E2、E3 六组要先实现并实测同样的 scoped-corpus 接口，然后才能开始 MedHop 主实验。评测值是候选答案的 normalized exact match，不是开放域临床诊断正确率；C 证据组单独评价，不可当作答题臂强行算准确率。旧 2,417 项实验调度不能无修改地用在 MedHop 上。

许可、论文、切分、限制见 `DATASET_CARD.md`。
