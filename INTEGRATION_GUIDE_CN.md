# GitHub Desktop 上传说明（只提交代码，不提交整个数据集）

1. 把这个 ZIP **解压后包含的文件与文件夹**复制到 Windows GitHub 仓库根目录：`C:\Users\99\Downloads\MultiHop-RAG-github`。
2. GitHub Desktop 中选择 `v13-agentic-rag-ablations` 分支。Changes 应出现 `MEDHOP_BENCHMARK.md`、`.github/workflows/medhop-adapter-check.yml`、`mhrag_v13_modular_starter/benchmarks/medhop/` 等新文件。
3. 确认 Changes 中不包含 `*.zip`、`*.parquet`、`*_gold_PRIVATE.*`、医学 PDF、原始数据库和任何 token。
4. Commit 标题：`Integrate MedHop benchmark metadata and HF fallback downloader`；再点 **Push origin**。
5. AutoDL 是不同机器。若你也想在 AutoDL 使用，需另行把 `benchmarks/medhop/` 文件夹上传到 AutoDL 项目目录；Windows 的 GitHub Desktop 提交**不会自动复制到 AutoDL**。

这是一个**可重复获取数据的公开项目集成**，不是把 340 MB QAngaroo 大文件强行放进 GitHub。MedHop 原始数据为 CC BY-SA 3.0，完整版权和引用请见 `DATASET_CARD.md`。
