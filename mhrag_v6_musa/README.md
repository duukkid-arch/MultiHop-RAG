# v6 750-character BM25 vs Hybrid, native MUSA edition

Designed for the AutoDL MTT S4000 image with **torch 2.2.0 + torch_musa 1.3.0** pre-installed. No CUDA and no torchada needed at first. This bundle contains an independent copy of the retrieval and CE scripts with explicit `musa` devices. Native MUSA support still needs validation; not all operators may work. `hybrid_musa.py` uses CPU NumPy for dense top-k of the small baseline corpus to avoid a potentially unsupported MUSA top-k operator. GPU remains responsible for BGE embedding and CE inference.

## Before starting
Copy existing project, original SQLite and dataset, v4 baseline files, local BGE model, and local reranker snapshot from NVIDIA instance. Existing v5 results are unnecessary. Check files using `python mhrag_v6_musa/run.py --dry-run`. Python environment should inherit or clone the AutoDL MUSA base, **never install CUDA torch**.

## Setup on new S4000
```
cd /root/autodl-tmp/MultiHop-RAG
unzip -o mhrag_v6_musa.zip
conda activate mhrag_musa
python mhrag_v6_musa/run.py --dry-run
python -u mhrag_v6_musa/smoke.py 2>&1 | tee /root/autodl-tmp/musa_smoke.log
```

If smoke succeeds:
```
cd /root/autodl-tmp/MultiHop-RAG
mkdir -p results/v6_focus_750_musa
nohup python -u mhrag_v6_musa/run.py --root results/v6_focus_750_musa >results/v6_focus_750_musa/launcher.log 2>&1 </dev/null &
echo $!
tail -f results/v6_focus_750_musa/launcher.log
```

You must evaluate CPU vs MUSA outputs numerically before publishing cross-hardware performance comparisons. Check logs if `STATUS=FAILED`. The experiment only measures document-level evidence metrics and uses 609 reconstructed article texts, not a large external corpus.
