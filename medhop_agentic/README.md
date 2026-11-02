# MedHop Agentic Retrieval — Reproducible CPU Baselines

This repository contains **two deterministic retrieval baselines** for the **original MedHop** candidate-selection task:

- `single`: single-hop TF-IDF retrieval with candidate evidence scoring.
- `agentic`: two-hop heuristic retrieval with bridge-document selection, query expansion and candidate scoring.

> **Research status:** This is **not** an LLM-controlled Agentic RAG system. The second mode follows a fixed two-hop heuristic. Full-dataset validation results are **not included** and no accuracy claim is made.

## Dataset

The expected MedHop original examples contain `query`, `supports` (list of passages), `candidates` (list), and `answer` (for evaluation). An optional `id` is used in saved predictions. The repository **does not redistribute the dataset**. Each question indexes its own provided `supports`; this is **not** corpus-wide retrieval.

On AutoDL, the dataset has been prepared at:

```text
/root/autodl-tmp/medhop_benchmark_private/medhop_original
```

The reported local counts are train **1,620** and validation **342**; counts are based on the prior setup log, not independently verified here.

## Requirements

Python 3.10+ recommended; GPU is not required for these baselines.

```bash
python -m pip install -r requirements.txt
```

## Full evaluation (342 validation examples per baseline)

```bash
bash run_full_experiment.sh
```

To override the dataset path:

```bash
bash run_full_experiment.sh /path/to/medhop_original
```

Both modes write predictions, traces, logs and metrics under `results/`. Example outputs:

```text
results/single_validation.jsonl
results/single_validation_metrics.json
results/agentic_validation.jsonl
results/agentic_validation_metrics.json
```

The metrics include exact-match accuracy **when gold `answer` labels are present**, the scored count and elapsed time. No results are prefilled.

## Individual commands

```bash
python medhop_pipeline.py --split validation --mode single --out results/single_validation.jsonl
python medhop_pipeline.py --split validation --mode agentic --out results/agentic_validation.jsonl
```

Set `--data` for a custom dataset location. `--limit 0` (default) means the entire split.

## Reproducibility and limitations

- TF-IDF is fit independently on the support texts of each question.
- Prediction is made without reading the gold `answer` field; the label is used in evaluation only.
- Two-hop mode uses heuristic bridge detection and document similarity, **not learned planning**.
- Different hardware and environment versions may change runtime. `requirements.txt` lists the project dependencies without pinned versions.
- Candidate-selection accuracy is not proof of faithful biomedical multi-hop reasoning. Inspect retrieved evidence and traces.
- Do not commit datasets, large model weights, credentials or private training artifacts. `.gitignore` excludes common directories, but **review `git status` before each commit**.

## Roadmap

1. Establish single-hop vs. heuristic two-hop validation baselines.
2. Add BM25 / dense retrieval baselines with consistent support-document context.
3. Implement LLM-controlled planning and tool use.
4. Add evidence validation, retrieval ablations and error analysis.
5. Report reproducible comparison tables with measured results.

## Files

- `medhop_pipeline.py` — data loading, retrieval, prediction and metrics.
- `run_full_experiment.sh` — full validation comparison in one command.
- `requirements.txt` — dependencies.
- `demo.json` — synthetic example for local inspection, **not** an official MedHop benchmark sample.
