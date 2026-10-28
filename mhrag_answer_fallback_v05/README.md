# MultiHop-RAG v0.5：安全原文摘录回退

**解决频繁出现“本地小模型生成了不可核查的回答，本次已屏蔽”而无法使用的问题。**

本补丁不会放行无引用的模型结论，也**不会自动在模型生成内容后加 `[1]` 伪造背书**。它保留原来 v0.2～v0.4 的证据校验。只有在模型未通过校验时，答案区自动切换到 **【原文摘录模式】**：显示最多两条来自检索结果的真实原文、对应的文献名称、PDF 页码和原有引用编号，并明确告诉用户这不是综合性医学结论。

这可以停止重复出现没有任何有用内容的拦截提示；不能单凭本补丁解决 1.5B 模型的医学推理能力。更高质量的综合回答仍需要更强模型与证据核验。

## 安装

上传 ZIP 到 `/root/autodl-tmp/MultiHop-RAG`，执行：

```bash
cd /root/autodl-tmp/MultiHop-RAG
unzip -o mhrag_answer_fallback_v05.zip
python mhrag_answer_fallback_v05/test_fallback.py
python mhrag_answer_fallback_v05/install.py
python -m py_compile mhrag_gpu_chat_musa/core.py mhrag_gpu_chat_musa/safe_evidence_fallback_v05.py
```

适用于已经安装 v0.2、v0.3 或 v0.4 的程序。安装器会检查 `core.py` 是否与预期匹配，自动备份，不改知识库和 GPU 文件。修改后需要重启 **7865 网页服务**，**8001 本地 Qwen 不变**。

建议重启前用 `ps -ef | grep '[m]hrag_gpu_chat_musa/app.py'` 确认 PID，再优雅终止网页进程（不是 Qwen）。在配置好的 screen 里重新启动网页程序。健康检查只代表连接状态，最终请通过真实问答检验结果是否出现【原文摘录模式】。

## 输出含义

- 通过引用和基础数值校验的模型答案：照常输出。
- 未通过引用核验的模型答案：不显示未经核查的模型结论，改显示带引用的**原文摘录**。
- 找不到证据：仍然提示知识库没有可核查来源。

**限制**：摘录仍可能不包含完整研究信息，可能截断语句，且无法验证因果结论、所有医学风险或原文证据之间的矛盾；对临床决策仍须查看原文与专业指南。
