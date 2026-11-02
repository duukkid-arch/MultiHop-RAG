# MedHop Benchmark Integration

This project supports reproducible preparation of the **QAngaroo MedHop** public biomedical multi-hop reasoning benchmark.

- [Dataset card, provenance, license and citation](mhrag_v13_modular_starter/benchmarks/medhop/DATASET_CARD.md)
- [中文复现与 AutoDL 下载说明](mhrag_v13_modular_starter/benchmarks/medhop/README_CN.md)
- [Source-preserving HF mirror downloader](mhrag_v13_modular_starter/benchmarks/medhop/download_medhop_hf.py)
- [Separated gold and question-scoped support corpus preparation](mhrag_v13_modular_starter/benchmarks/medhop/prepare_medhop.py)
- [Prediction scorer](mhrag_v13_modular_starter/benchmarks/medhop/score_medhop.py)

**Data not vendored:** the approximately 340 MB official QAngaroo archive is stored by its original publisher. This repository includes the implementation and a verifiable workflow rather than duplicating large binary files. The MedHop dataset itself is distributed under **CC BY-SA 3.0**; cite Welbl et al. (TACL, 2018).

**Scope:** official public train/validation benchmark, not a newly expert-reviewed private holdout. GPU algorithm integration and six-arm benchmark runs have not yet been verified.
