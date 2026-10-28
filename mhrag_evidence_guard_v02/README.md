# MultiHop-RAG Evidence Guard v0.2

A conservative hotfix to **existing** `mhrag_gpu_chat_musa` v0.1, not a new RAG system.

Fixes:
- Gate explicit clozapine questions against snippets mentioning clozapine/氯氮平. Unrelated depression/TAVR papers will not be accepted as evidence.
- Gate explicit depression-etiology questions against snippets with clear etiology context, preventing treatment-only snippets from passing as causes.
- Default 4 snippets * 650 chars to prevent context overflow in local Qwen2.5-1.5B.
- Block answers with out-of-range, excessively repeated or missing citations. Does **not** guarantee all supported citations are factually correct.
- Keep original file/DB/indices, create timestamped backup of core.py and llm.py.

Install after stopping only the port-7865 server (leave port-8001 Qwen running):

```bash
cd /root/autodl-tmp/MultiHop-RAG
unzip -o mhrag_evidence_guard_v02.zip
python mhrag_evidence_guard_v02/test_guard.py
python mhrag_evidence_guard_v02/install.py
python -m unittest discover -s mhrag_gpu_chat_musa/tests -v
```

Restart the 7865 server with the same LLM env variables.

**Caveats:** The topic gate is heuristic and covers the two observed failures, not all possible medical questions. It may falsely reject a truly relevant excerpt, and some unsupported statements could still slip through. Manual checking of paper titles, passages and pages remains necessary. This is a stopgap; future improvements should add independently verified relevance and entailment/claim checking, calibrate thresholds with gold labels, and cite claim-specific evidence.
