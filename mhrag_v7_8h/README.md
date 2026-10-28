# MultiHop-RAG v7: 8-hour capped experiment **after / alongside existing v6**

## Rule zero
Do **not** start v7 GPU benchmarking while v6 is running. v6 scores each of two pools for 2,255 labeled questions × up to 100 candidates = up to 451,000 CE pairs. The current v6 launcher has no hard 8-hour cap; track the elapsed wall time from its original start. It checkpoints rerank scores per question, and can be resumed with the same root after stopping it cleanly if needed. Do not delete score files or edit the v6 root while it runs.

## Stage 1: finish current v6

In MTT S4000:

```bash
cd /root/autodl-tmp/MultiHop-RAG
R=results/v6_focus_750_musa
cat "$R/STATUS"
tail -20 "$R/launcher.log"
tail -12 "$R/ce_bm25_750.log" 2>/dev/null || true
tail -12 "$R/ce_hybrid_750.log" 2>/dev/null || true
```

If v6 doesn't finish within rental time, prioritize checkpoint integrity. Do not launch a new GPU workload.

## Stage 2: CPU-only budget curve and gold-family cluster bootstrap

Copy the entire `mhrag_v7_8h` directory to `/root/autodl-tmp/MultiHop-RAG/` and run:

```bash
cd /root/autodl-tmp/MultiHop-RAG
python mhrag_v7_8h/analyze_cached.py --bootstrap 1000 --tolerance-pp 0.5
cat results/v7_8h_budget/V7_BUDGET_REPORT.json
```

This does not load Transformers or rescore. It needs `eval_bm25_750/per_query.csv` and/or `eval_hybrid_750/per_query.csv`. DEV selects a CE candidate K in {20,30,60,100} per oracle question type, choosing the cheapest K whose DEV All-Gold is within 0.5 percentage points of the best DEV K. HOLDOUT is used only for reporting. It additionally reports cluster-bootstrap confidence intervals that group exactly identical gold evidence sets; overlapping but nonidentical evidence sets are not grouped. **Caveat**: v5 holdout has already been inspected repeatedly, so this cannot be called a fresh blind validation. True question types are oracle labels, not classifier predictions. CE@K values are retrospective simulations from 100-score caches.

## Stage 3: real CE scoring wall-time measurement on MUSA

Run only if v6 `STATUS=DONE`, and run under screen or nohup:

```bash
cd /root/autodl-tmp/MultiHop-RAG
screen -S v7_latency
HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 python -u mhrag_v7_8h/benchmark_musa.py --sample 40 --max-seconds 2700 \
  2>&1 | tee results/v7_8h_budget/latency.log
```

The stopwatch excludes the first warmup pass; it measures MUSA CE model inference of the same stratified questions at K=60 and K=100, with order alternated. It saves each query's latency, is append-only, and stops after up to 45 minutes of active scoring (a single query may slightly overshoot). Model load and preprocessing overhead are not included in the timed per-query values. No v6 scores are modified. Output `results/v7_8h_budget/latency/latency.jsonl`.

## Time allocation for an 8h **total** time window

- Hours 0–4: allow the already-running v6 to progress; monitor CE throughput. If slower than expected, do not begin v7 GPU work.
- Hour 4–5: CPU analysis and paired comparisons using any completed evaluations.
- Hour 5–6: oracle-type budget frontier and evidence-family bootstrap (CPU only).
- Hour 6–7: optional <=45min actual MUSA K=60 vs 100 benchmark, **only after v6 DONE**.
- Hour 7–8: error audit, write conclusions, save logs and generated score files.

The hours are a planning reserve, not a claim about hardware throughput. If v6 consumes the whole eight hours, do only v6 and postpone optional stages. Shut down normally after all files are flushed; do not run a second v6 instance.

## Research claims

1. Original v4 strong baseline versus BM25/Hybrid on 750-character reconstructed chunks, document-level All-Gold@10 and Coverage@10.
2. Paired quality versus theoretical CE candidate count, using cached scores; these are NOT observed speedups.
3. DEV-selected question-type budget strategies are ORACLE-type policies unless you build and evaluate a real question-type classifier.
4. Grouped uncertainty handles exact duplicate gold sets but NOT overlapping semantic families or independent external test validation.
5. MUSA latency results are measured on an actual MTT S4000, but they are CE-only times, not whole-pipeline latency.
