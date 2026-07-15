# Old Stories, New Readers — Reproduction Code

Code to reproduce the retrieval and QA results in *Old Stories, New Readers:
Evaluating Retrieval and QA over Historical Newspapers under Diachronic Vocabulary
Change* (JCDL 2026): BM25 / dense (E5‑FT) / hybrid retrieval, the fusion‑weight
selection with significance testing, and end‑to‑end RAG‑QA.

## Layout

```
hybrid_eval.py        BM25 + dense fusion (min–max), metrics, --dump_runs
alpha_experiment.py   fusion-weight (alpha) selection: sweep, bootstrap CIs,
                      paired significance vs. best single retriever, oracle bound
retrieval_cache.py    load_artifacts(): builds/caches BM25 index + E5-FT corpus
                      embeddings (imported by eval_three / qa_all)
eval_three.py         BM25 / E5-FT / Hybrid at a fixed alpha -> R@1/10/100, nDCG, MRR
qa_all.py             one-shot end-to-end QA: hybrid top-k retrieval + QLoRA reader
qa_eval_runs.py       QA reader eval over pre-dumped runs (LLM or extractive)
models/datr.py        dual-encoder used for the E5-FT checkpoint (era-aware; run --no_era)
utils/helpers.py      config / io / seed helpers
configs/config.yaml   DATR config (BERT-base)
configs/config_e5.yaml  E5-FT config (encoder intfloat/e5-base-v2; use --no_era)
```

## Setup

```bash
conda create -n nros python=3.11 -y && conda activate nros
pip install -r requirements.txt
```

## Inputs you must provide

These are large and are **not** bundled here:

- **Corpus + questions** — `data/processed/corpus.jsonl`, `test_retrieval.jsonl`,
  `dev_retrieval.jsonl` from **ChroniclingAmericaQA**
  (https://github.com/DataScienceUIBK/ChroniclingAmericaQA).
- **Modern queries** — the companion `../dataset/modern_queries_test.jsonl`
  (rename/symlink to `data/processed/test_retrieval_modern.jsonl`). The dev-split
  modern queries used for `--tune_file` are available on request; without it,
  `alpha_experiment.py` tunes on the eval set.
- **Checkpoints** — the fine‑tuned dense retriever `outputs_e5_ft/best_model.pt`
  and the QLoRA reader adapter `outputs/qlora_llama/best_adapter`
  (base `meta-llama/Llama-3.1-8B-Instruct`). Available from the authors on request.

(The E5‑FT config, `configs/config_e5.yaml`, and the retriever code in `models/`
are bundled; only the large corpus and checkpoints above are external.)

## Reproduce

**1. Retrieval table + alpha selection + significance + oracle** (Table: hybrid):

```bash
python alpha_experiment.py \
  --config configs/config_e5.yaml --datr_checkpoint outputs_e5_ft/best_model.pt --no_era \
  --corpus  data/processed/corpus.jsonl \
  --eval    data/processed/test_retrieval_modern.jsonl \
  --tune_file data/processed/dev_retrieval_modern.jsonl \
  --device cuda --boot 1000
```

Prints the report‑set metrics (R@1/5/10/100, nDCG, MRR for BM25 / E5‑FT / Hybrid),
the selected `alpha*`, MRR 95% CIs, the paired bootstrap vs. the better single
retriever, and the per‑query oracle MRR. (`eval_three.py` is an auxiliary
cross‑check that ranks over the full corpus with `bm25s`; its BM25 differs slightly
from the stemmed BM25 used in the paper.)

**2. End‑to‑end QA.** Either the one‑shot script:

```bash
python qa_all.py \
  --config configs/config_e5.yaml --ckpt outputs_e5_ft/best_model.pt \
  --corpus  data/processed/corpus.jsonl \
  --queries data/processed/test_retrieval.jsonl \
  --queries_modern data/processed/test_retrieval_modern.jsonl \
  --no_era --device cuda --max_samples 7000   # EM/F1/Recall/Precision, oracle + hybrid
```

…or the two‑step canonical path (dump runs, then read):

```bash
python hybrid_eval.py --dense datr --datr_checkpoint outputs_e5_ft/best_model.pt \
  --config configs/config_e5.yaml --no_era \
  --corpus data/processed/corpus.jsonl \
  --queries data/processed/test_retrieval.jsonl \
  --queries_modern data/processed/test_retrieval_modern.jsonl \
  --alpha 0.60 --cand 1000 --dump_runs --dump_topk 20 --output_dir outputs/qa_runs

python qa_eval_runs.py --reader llm --runs outputs/qa_runs/runs_modern.jsonl \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --adapter_path outputs/qlora_llama/best_adapter \
  --ctx_k 10 --output outputs/qa/llm_ft_modern.json
```

## Notes
- Pass `--no_era` everywhere with the E5‑FT checkpoint: it was trained without era
  conditioning, so era embeddings must be disabled at inference.
- 4‑bit reader: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and, on a
  24 GB GPU, keep `qa_all.py`'s `--gen_batch` small (default 4).

## License
MIT (see `LICENSE`). Data is released separately under CC‑BY‑4.0 (see `../dataset/`).
See `../dataset/README.md` for citations (this paper + ChroniclingAmericaQA).
