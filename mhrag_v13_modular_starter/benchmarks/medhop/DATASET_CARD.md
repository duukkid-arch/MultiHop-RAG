# Dataset Card — QAngaroo MedHop

| Field | Value |
|---|---|
| Name | **MedHop**, QAngaroo multi-hop reading comprehension benchmark |
| Authors | Johannes Welbl, Pontus Stenetorp, Sebastian Riedel |
| Publication | *Constructing Datasets for Multi-hop Reading Comprehension Across Documents*, TACL 2018 |
| DOI | [10.1162/tacl_a_00021](https://doi.org/10.1162/tacl_a_00021) |
| Homepage | https://qangaroo.cs.ucl.ac.uk/ |
| Official archive | https://zenodo.org/records/6407402 |
| Alternative access | https://huggingface.co/datasets/bigbio/medhop (`medhop_source` parquet) |
| License (dataset) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| Public labelled splits | Train **1,620**; validation **342** |
| Task | Drug/protein interaction, multi-document, multiple-choice reasoning |

## What this repository does / does not ship

The Git repository versions **adapters, download instructions, validation checks, dataset metadata, citations and tests**, not the binary QAngaroo archive or raw PubMed evidence documents. The official QAngaroo archive is approximately 339.8 MB, includes WikiHop, and exceeds GitHub's ordinary per-file hard limit. The download script selects only MedHop source-format Parquet from BigBio's public converted representation. Availability depends on the network and upstream server.

Do not confuse public MedHop validation gold with a novel private holdout; neither MedHop nor its support passages constitute the original 116-PDF corpus. This benchmark tests question-scoped reading of supplied support paragraphs, not unrestricted open-corpus medical retrieval. No clinical approval, original gold annotation, medical diagnosis, or expert re-review is claimed.

Each example contains `id`, `query`, `candidates`, `supports` (a list of documents), and `answer`. The transformation preserves each `supports` item and puts `answer` in a separate **offline scoring file**. The gold file is publicly known and separated to prevent accidental exposure to model prompts, not to imply secrecy of the public labels. The model may see candidates, as MedHop is a candidate-selection task.

## Citation

```bibtex
@article{welbl-etal-2018-constructing,
  title={Constructing Datasets for Multi-hop Reading Comprehension Across Documents},
  author={Welbl, Johannes and Stenetorp, Pontus and Riedel, Sebastian},
  journal={Transactions of the Association for Computational Linguistics},
  volume={6},
  pages={287--302},
  year={2018},
  doi={10.1162/tacl_a_00021}
}
```

Data license: **Creative Commons Attribution-ShareAlike 3.0 Unported**. If you redistribute data or adapted excerpts, retain appropriate attribution, indicate changes, and observe share-alike terms. Your source-code license may be different; this dataset notice is not a license grant for all repository code.
