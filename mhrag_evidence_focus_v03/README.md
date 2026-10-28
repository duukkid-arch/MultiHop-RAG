# MultiHop-RAG Evidence-First v0.3

For the existing MUSA GPU chat v0.1 **after** the v0.2 evidence guard is installed.

Addresses the observed case where the local Qwen2.5-1.5B misread a historical introduction paragraph in WPS-24-250.pdf and ignored a passage reporting an aHR and 95% CI. **This is not clinical grade or a factuality guarantee.**

- Prioritizes numerical *results* evidence over prior-work introduction text **without changing citation IDs**.
- Uses up to 1,150 characters per source when there are at most 2 sources; v0.2 truncated at 650 characters.
- Concise evidence-only prompt. No chat history by default (opt in with `CHAT_ANSWER_HISTORY=1`).
- Blocks a specific dangerous failure mode: claiming no reported clinical effect where retrieved evidence has an aHR and CI. Other unsupported claims may still slip through.
- Backs up `core.py` and `llm.py`, does not touch GPU, model weights, or `knowledge.sqlite`.

## Install

Stop **only port 7865** chat service. Keep port 8001 Qwen server running. From `/root/autodl-tmp/MultiHop-RAG`:

```bash
unzip -o mhrag_evidence_focus_v03.zip
python mhrag_evidence_focus_v03/test_focus.py
python mhrag_evidence_focus_v03/install.py
python mhrag_evidence_focus_v03/test_focus.py
python -m unittest discover -s mhrag_gpu_chat_musa/tests -v
```

Start the chat service with the usual environment variables (`LLM_BASE_URL=http://127.0.0.1:8001/v1`, `LLM_MODEL=qwen-local`, `LLM_API_KEY=local-only`, `CHAT_ANSWER_SOURCES=4`), in a `screen`. Only one instance should bind 7865.

## Notes

- The v0.2 topic filter is still rule-based; source ranking and text generation require human validation.
- For articles with truncated numerical results, the next structural improvement is fetching adjacent chunks within the same document/page in the DB or re-chunking PDFs with overlap that keeps statistical statements intact; this patch does not change chunks.
- Larger models may improve performance if compatible with the current MUSA runtime, but do not replace evidence-grounding checks.
