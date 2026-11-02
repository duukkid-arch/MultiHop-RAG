# MultiHop-RAG: Agentic Multi-Hop Retrieval and Reasoning

**An experimental framework for Agentic Retrieval-Augmented Generation (Agentic RAG), multi-hop evidence retrieval, and reproducible reasoning benchmarks.**

This repository investigates how retrieval agents can **plan searches, follow evidence across documents, verify answers, and evaluate reasoning trajectories**. The immediate focus is the original **MedHop** benchmark, where a model must select the correct answer from candidates using supporting passages.

> **Current implementation status:** The included MedHop `agentic` mode is a **heuristic two-hop retrieval baseline**, not yet an LLM-directed autonomous agent. LLM planning, tool selection, and evidence verification are planned extensions. Benchmark scores are reported only after experiments have been run.

## Why Agentic RAG?

Ordinary RAG typically retrieves evidence once and then predicts an answer. Agentic RAG aims to adapt the retrieval process to the evidence already seen:

1. **Understand the task** and identify entities and candidate answers.
2. **Retrieve first-hop evidence** relevant to the original question.
3. **Follow a bridge** by selecting intermediate passages or entities.
4. **Retrieve additional evidence** to connect the starting entity to a candidate answer.
5. **Verify and answer** using the evidence path; record the retrieval trace for inspection.

The current heuristic implementation approximates steps 2–4 with deterministic TF-IDF and similarity scoring. Later experiments will replace fixed heuristics with explicit LLM-based planning and verification.

## Repository components

| Component | Purpose | Status |
| --- | --- | --- |
| `mhrag_v13_modular_starter/` | Existing modular retrieval and benchmark infrastructure | Existing project code; see its own documentation |
| `mhrag_v13_modular_starter/benchmarks/medhop/` | MedHop benchmark integration and offline checks | Existing project code |
| `medhop_agentic/medhop_pipeline.py` | Per-question retrieval and candidate-answer scoring | Implemented baseline |
| `medhop_agentic/run_full_experiment.sh` | Full validation comparison of two retrieval modes | Implemented script |
| `medhop_agentic/README.md` | Detailed dataset assumptions, commands, and limitations | Documentation |

Directory names above describe the intended integrated repository layout; check your branch for the available files.

## MedHop experiment

**Task.** Given `query`, `supports`, and `candidates`, predict a candidate answer; use the gold `answer` field for evaluation only. Support passages are retrieved **within each question's provided documents**, not from a shared global corpus.

**Baseline A — Single-hop retrieval (`single`)**  
Uses TF-IDF relevance to retrieve supporting passages and rank candidate answers.

**Baseline B — Heuristic two-hop retrieval (`agentic`)**  
Uses first-hop evidence to identify bridge passages, expands retrieval based on similar documents, and scores candidate answers with evidence from multiple passages. This is an *agentic-style retrieval baseline*; it does not autonomously call an LLM or choose its own tools.

### Run full validation on AutoDL

The following commands assume the `medhop_agentic/` directory has been added to this repository and the MedHop dataset is available locally.

```bash
cd medhop_agentic
python -m pip install -r requirements.txt
bash run_full_experiment.sh /root/autodl-tmp/medhop_benchmark_private/medhop_original
```

By default, the script evaluates the entire MedHop **validation** split in both modes. The previously prepared local dataset was reported as **1,620 training** and **342 validation** examples; these counts are environment-specific and are not reverified by this README.

### Outputs

```text
medhop_agentic/results/
  single_validation.jsonl
  single_validation_metrics.json
  agentic_validation.jsonl
  agentic_validation_metrics.json
```

Each prediction record contains an answer, gold label (when present), and retrieval trace. Metric JSON files contain scored count, exact-match accuracy when labels exist, and elapsed time. **No measured accuracy is claimed in advance.**

## Agentic research roadmap

- **Stage 1 — Reproducible baselines:** single-hop versus deterministic two-hop retrieval; inspect failures and retrieval traces.
- **Stage 2 — Retrieval upgrades:** BM25 and dense retrieval under consistent candidate sets and support-document conditions.
- **Stage 3 — LLM agent:** planner decides whether to retrieve again, selects search actions, and terminates when evidence is sufficient.
- **Stage 4 — Evidence verification:** assess support-chain consistency and reject unsupported answers.
- **Stage 5 — Ablations and reporting:** compare accuracy, latency, retrieval steps, evidence coverage, and compute cost.

The roadmap is a **plan**, not a statement that all stages are already implemented.

## Reproducibility and responsible data handling

- Run evaluation on held-out validation examples and do not use the gold answer for prediction.
- Preserve retrieval traces and experiment configuration for ablation studies.
- Do not commit private datasets, access tokens, model checkpoints, or large output artifacts.
- Consult the MedHop dataset's own terms and citation guidance before redistribution or publication.

## Development

Use feature branches and pull requests for experimental changes. Keep new approaches comparable against the same validation split, candidate answers, and evaluation metric.

<!-- AGENTIC_ARCHITECTURE_START -->
## Agentic RAG Architecture / 椤圭洰杩愯鍘熺悊

涓嬮潰灞曠ず鏈」鐩殑**姒傚康鎬ц繍琛屾祦绋?*銆傚叾涓绱€佽瘉鎹暣鍚堝拰绛旀楠岃瘉鏄綋鍓嶄唬鐮佺殑涓昏妯″潡锛汱LM 瑙勫垝鍜屽姩鎬佸伐鍏疯皟搴﹀睘浜庢寔缁畬鍠勭殑 Agentic 鑳藉姏锛?*骞堕潪鎵€鏈夊疄楠屽垎鏀兘宸插疄鐜板畬鏁磋嚜涓昏鍒?*銆?
```mermaid
flowchart TD
    Q[鐢ㄦ埛闂 / Evaluation Query] --> D[闂瑙ｆ瀽涓庡瓙鐩爣鍒嗚В]
    D --> P[妫€绱㈢瓥鐣ラ€夋嫨 / Agentic Planning]
    P --> R[鍊欓€夋枃妗ｆ绱?/ Retrieval]
    R --> I[鐮旂┒韬唤涓庢潵婧愯矾鐢?/ Study Routing]
    I --> H[璇佹嵁鏀堕泦涓庣墖娈垫暣鐞?/ Evidence Harvesting]
    H --> G[鍩轰簬璇佹嵁鐨勫€欓€?Claim 鐢熸垚]
    G --> V{寮曟枃涓庢暟鍊奸妫€閫氳繃?}
    V -- 鍚?--> A[寮冪瓟鎴栬褰曡瘉鎹笉瓒砞
    V -- 鏄?--> C{闂瑕嗙洊鍙婂畨鍏ㄦ潯浠舵弧瓒?}
    C -- 鍚?--> A
    C -- 鏄?--> O[杈撳嚭鏈夋潵婧愮害鏉熺殑鍊欓€夌瓟妗圿
    O --> E[瀹為獙璇勪及涓庢秷铻嶅垎鏋怾
    A --> E
    E --> L[璁板綍鍑嗙‘鐜囥€佽鐩栫巼銆佸純绛旂巼涓庤€楁椂]

    classDef process fill:#edf4ff,stroke:#5487ce,color:#172d4f;
    classDef gate fill:#fff6df,stroke:#be8a22,color:#594008;
    classDef stop fill:#fff0f0,stroke:#b85a5a,color:#5c2020;
    class D,P,R,I,H,G,O,E,L process;
    class V,C gate;
    class A stop;
```

**濡備綍闃呰锛?* 绯荤粺鍏堣В鏋愰棶棰樸€佹绱㈠苟鏍稿鐮旂┒鏉ユ簮锛屽啀浠庤鏂囪瘉鎹墖娈电敓鎴愬€欓€?Claim銆傚彧鏈夐€氳繃寮曟枃/鏁板€煎垵妫€涓庨棶棰樿鐩栨鏌ワ紝鍊欓€夊洖绛旀墠浼氳杈撳嚭锛涜瘉鎹笉瓒虫椂璁板綍鍘熷洜骞跺純绛斻€傝嚜鍔ㄦ鏌?*涓嶇瓑浜庡尰瀛︿笓瀹朵簨瀹炴牳楠?*銆?
### 浠ｇ爜涓庡疄楠屽叆鍙ｏ紙鎸夊姛鑳斤級

| 鍔熻兘 | 鐩稿叧妯″潡 / 鐩綍 | 浣滅敤 |
| --- | --- | --- |
| Agentic 宸ヤ綔娴佷笌璺敱 | `mhrag_v13/grounded_controller.py` | 缁勭粐妫€绱€佹潵婧愯矾鐢便€佺敓鎴愪笌鏈€缁堝喅绛?|
| 澶氱粍娑堣瀺绛栫暐 | `mhrag_v13/ablation_arms.py` | 姣旇緝涓嶅悓鐢熸垚涓庡畨鍏ㄧ瓥鐣?|
| 璇佹嵁寮曠敤鏍稿 | `mhrag_v13/e2_quote_probe.py` | 鍘熸枃閫愬瓧寮曟枃鍊欓€夋鏌?|
| 瀛楁绾?Claim 鐢熸垚 | `mhrag_v13/e2_field8.py` | 鐢熸垚鐩爣瀛楁鐩稿叧鍊欓€?Claim |
| 闂瑕嗙洊璇勪及 | `mhrag_v13/question_coverage.py` | 妫€鏌ュ€欓€変簨瀹炴槸鍚﹁鐩栨彁闂?|
| MedHop 瀹為獙锛堣嫢宸查儴缃诧級 | `medhop_agentic/` | 鍗曡烦/鍙岃烦涓?BM25 鐩稿叧瀹為獙鍏ュ彛 |

> **鐮旂┒鐘舵€佽鏄庯細** 褰撳墠椤圭洰鍖呭惈鍚彂寮忓璺虫绱㈠熀绾垮強 Agentic/璇佹嵁楠岃瘉瀹為獙浠ｇ爜銆傚浘涓殑鈥淎gentic Planning鈥濅唬琛ㄧ洰鏍囪兘鍔涙垨閮ㄥ垎绛栫暐锛屼笉搴旂悊瑙ｄ负鎵€鏈夊疄楠岄兘宸茬粡鍏峰缁忚繃楠岃瘉鐨?LLM 鑷富宸ュ叿璋冪敤銆侻edHop 鍑嗙‘鐜囦笌鍖诲鏂囩尞 Claim 楠岃瘉鐜囧睘浜庝笉鍚岃瘎娴嬶紝涓嶅簲娣蜂负涓€璋堛€?<!-- AGENTIC_ARCHITECTURE_END -->
