"""
Hybrid retrieval evaluation: BM25 + a chosen dense retriever, with the paper's
min-max score fusion (alpha * BM25 + (1-alpha) * dense). Reports R@1/5/10/100 + MRR.

Purpose (the decisive comparison): measure Hybrid(BM25 + DATR) vs
Hybrid(BM25 + E5/BGE/GTE/Contriever) on identical footing, to see whether the
era-aware DATR actually earns its place over an off-the-shelf dense model.

Run from the project root (so `models/` and `utils/` import).

Dense options:
  --dense datr   (requires --datr_checkpoint and --config; era-aware encoding)
  --dense e5 | bge | gte | contriever   (off-the-shelf HF encoder)

Examples
--------
  # Hybrid with your DATR
  python hybrid_eval.py --dense datr --datr_checkpoint outputs/best_model.pt \
     --config configs/config.yaml \
     --corpus data/processed/corpus.jsonl --queries data/processed/test_retrieval.jsonl \
     --queries_modern data/processed/test_retrieval_modern.jsonl --alpha 0.3

  # Hybrid with off-the-shelf E5
  python hybrid_eval.py --dense e5 \
     --corpus data/processed/corpus.jsonl --queries data/processed/test_retrieval.jsonl \
     --queries_modern data/processed/test_retrieval_modern.jsonl --alpha 0.3

  # quick logic canary (corpus still fully encoded; limits queries)
  python hybrid_eval.py --dense e5 --max_queries 500 --corpus ... --queries ...
"""

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

HF = {  # name, pooling, qprefix, pprefix
    "e5": ("intfloat/e5-base-v2", "mean", "query: ", "passage: "),
    "bge": ("BAAI/bge-base-en-v1.5", "cls",
            "Represent this sentence for searching relevant passages: ", ""),
    "gte": ("thenlper/gte-base", "mean", "", ""),
    "contriever": ("facebook/contriever", "mean", "", ""),
}


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------- dense encoders ----------------------------------
class HFEncoder:
    def __init__(self, model_name, pooling, device):
        self.pooling = pooling
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def _pool(self, h, m):
        if self.pooling == "cls":
            return h[:, 0]
        m = m.unsqueeze(-1).expand(h.size()).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts, bs, max_len):
        out = []
        for i in tqdm(range(0, len(texts), bs), desc="enc"):
            enc = self.tok(texts[i:i+bs], padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(self.device)
            e = self._pool(self.model(**enc).last_hidden_state, enc["attention_mask"])
            out.append(torch.nn.functional.normalize(e, p=2, dim=-1).cpu().numpy())
        return np.vstack(out).astype("float32")


class DATRWrapper:
    """Encoding using the project's DATR model. If no_era=True, era conditioning
    is disabled (era_ids=None) -- required for checkpoints trained with --no_era,
    e.g. the E5 domain fine-tune."""
    def __init__(self, checkpoint, config_path, device, no_era=False):
        from utils.helpers import load_config
        from models.datr import DATR
        self.cfg = load_config(config_path)
        self.no_era = no_era
        bins = self.cfg["eras"]["bins"]
        self.era_to_id = {e["name"]: i for i, e in enumerate(bins)}
        self.era_to_id["unknown"] = len(bins)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(self.cfg["model"]["encoder_name"])
        self.model = DATR(
            encoder_name=self.cfg["model"]["encoder_name"],
            era_embedding_dim=self.cfg["model"]["era_embedding_dim"],
            num_eras=len(bins), pooling=self.cfg["model"]["pooling"],
            normalize=self.cfg["model"]["normalize_embeddings"], shared_encoder=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.to(device).eval()

    @torch.no_grad()
    def _encode(self, texts, eras, which, bs, max_len):
        out = []
        for i in tqdm(range(0, len(texts), bs), desc=f"datr-{which}"):
            bt = texts[i:i+bs]
            be = eras[i:i+bs]
            enc = self.tok(bt, padding=True, truncation=True, max_length=max_len,
                           return_tensors="pt").to(self.device)
            era_ids = None if self.no_era else torch.tensor(
                [self.era_to_id.get(e, self.era_to_id["unknown"]) for e in be],
                device=self.device)
            if which == "passage":
                emb = self.model.encode_passages(enc["input_ids"], enc["attention_mask"], era_ids)
            else:
                emb = self.model.encode_queries(enc["input_ids"], enc["attention_mask"], era_ids)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype("float32")

    def encode_corpus(self, corpus, bs):
        ml = self.cfg["training"]["max_seq_length"]
        return self._encode([p["text"] for p in corpus],
                            [p.get("era", "unknown") for p in corpus], "passage", bs, ml)

    def encode_queries(self, queries, bs):
        ml = self.cfg["training"]["max_seq_length"] // 4
        # queries get the 'unknown' era, matching the paper / evaluate.py
        return self._encode(queries, ["unknown"] * len(queries), "query", bs, ml)


# ---------------------------- retrieval helpers -------------------------------
def dense_topn(corpus_emb, q_emb, n):
    try:
        import faiss
        idx = faiss.IndexFlatIP(corpus_emb.shape[1]); idx.add(corpus_emb)
        s, i = idx.search(q_emb, n); return i, s
    except Exception:
        sims = q_emb @ corpus_emb.T
        part = np.argpartition(-sims, n-1, axis=1)[:, :n]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        i = part[rows, order]
        return i, sims[rows, i]


def bm25_topn(corpus_texts, queries, n):
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english")
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(corpus_texts, stemmer=st))
    qt = bm25s.tokenize(queries, stemmer=st)
    res = bm.retrieve(qt, k=n)
    return res.documents, res.scores  # [Q,n] corpus indices, scores


def minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    vs = list(d.values()); lo, hi = min(vs), max(vs)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def fuse_and_score(bm_docs, bm_scores, dn_idx, dn_scores, corpus_ids,
                   positives, alpha, ks):
    Q = len(positives)
    metrics = {f"recall@{k}": 0 for k in ks}; mrr = 0.0
    for qi in range(Q):
        bm = {corpus_ids[int(bm_docs[qi][j])]: float(bm_scores[qi][j])
              for j in range(len(bm_docs[qi]))}
        dn = {corpus_ids[int(dn_idx[qi][j])]: float(dn_scores[qi][j])
              for j in range(len(dn_idx[qi]))}
        bmn, dnn = minmax(bm), minmax(dn)
        fused = {}
        for pid in set(bmn) | set(dnn):
            fused[pid] = alpha * bmn.get(pid, 0.0) + (1 - alpha) * dnn.get(pid, 0.0)
        ranked = [pid for pid, _ in sorted(fused.items(), key=lambda x: -x[1])]
        pos = positives[qi]
        for k in ks:
            if pos in ranked[:k]:
                metrics[f"recall@{k}"] += 1
        if pos in ranked:
            mrr += 1.0 / (ranked.index(pos) + 1)
    out = {k: 100.0 * v / Q for k, v in metrics.items()}
    out["mrr"] = 100.0 * mrr / Q
    return out


def fuse_topk_runs(bm_docs, bm_scores, dn_idx, dn_scores, corpus_ids, alpha, topk):
    """Return, per query, the top-k fused passage ids (same fusion as scoring)."""
    Q = len(bm_docs)
    runs = []
    for qi in range(Q):
        bm = {corpus_ids[int(bm_docs[qi][j])]: float(bm_scores[qi][j])
              for j in range(len(bm_docs[qi]))}
        dn = {corpus_ids[int(dn_idx[qi][j])]: float(dn_scores[qi][j])
              for j in range(len(dn_idx[qi]))}
        bmn, dnn = minmax(bm), minmax(dn)
        fused = {}
        for pid in set(bmn) | set(dnn):
            fused[pid] = alpha * bmn.get(pid, 0.0) + (1 - alpha) * dnn.get(pid, 0.0)
        ranked = [pid for pid, _ in sorted(fused.items(), key=lambda x: -x[1])][:topk]
        runs.append(ranked)
    return runs


def run(corpus, corpus_ids, corpus_texts, eval_data, qfield, dense, args, device, tag=None):
    """Compute BM25 + dense candidates once, then fuse at every alpha in args.alphas.
    Returns {alpha: metrics}. If args.dump_runs, also writes top-k passages per query."""
    queries = [r[qfield] for r in eval_data]
    positives = [r["positive_passage_id"] for r in eval_data]
    # BM25 (raw text for BM25)
    bm_docs, bm_scores = bm25_topn(corpus_texts, queries, args.cand)
    # dense
    if args.dense == "datr":
        q_emb = dense.encode_queries(queries, args.batch_size)
    else:
        name, pool, qpref, ppref = HF[args.dense]
        q_emb = dense.encode([qpref + q for q in queries], args.batch_size, 64)
    dn_idx, dn_scores = dense_topn(args._corpus_emb, q_emb, args.cand)
    out = {}
    for a in args.alphas:
        out[a] = fuse_and_score(bm_docs, bm_scores, dn_idx, dn_scores, corpus_ids,
                                positives, a, args.ks)
    if args.dump_runs:
        a = args.dump_alpha if args.dump_alpha is not None else args.alphas[0]
        ranked_lists = fuse_topk_runs(bm_docs, bm_scores, dn_idx, dn_scores,
                                      corpus_ids, a, args.dump_topk)
        id_to_text = {cid: t for cid, t in zip(corpus_ids, corpus_texts)}
        runs_path = os.path.join(args.output_dir, f"runs_{tag or qfield}.jsonl")
        with open(runs_path, "w") as fo:
            for r, pids in zip(eval_data, ranked_lists):
                fo.write(json.dumps({
                    "query": r[qfield],
                    "answer": r.get("answer", ""),
                    "positive_passage_id": r["positive_passage_id"],
                    "retrieved": [{"passage_id": pid, "text": id_to_text.get(pid, "")}
                                  for pid in pids],
                }) + "\n")
        print(f"  dumped top-{args.dump_topk} runs (alpha={a}) -> {runs_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="datr|e5|bge|gte|contriever")
    ap.add_argument("--datr_checkpoint", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no_era", action="store_true",
                    help="disable era conditioning (use for --no_era checkpoints, e.g. E5-FT)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--queries_modern", default=None)
    ap.add_argument("--alpha", type=float, default=0.3, help="weight on BM25")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="sweep these BM25 weights (e.g. 0.3 0.5 0.7 0.9); overrides --alpha")
    ap.add_argument("--cand", type=int, default=1000, help="candidate depth per retriever")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_passage_len", type=int, default=256)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--output_dir", default="outputs/hybrid")
    ap.add_argument("--dump_runs", action="store_true",
                    help="also write top-k retrieved passages per query (for QA reader eval)")
    ap.add_argument("--dump_alpha", type=float, default=None,
                    help="alpha at which to dump runs (default: first of --alphas)")
    ap.add_argument("--dump_topk", type=int, default=5, help="passages per query to dump")
    args = ap.parse_args()
    if args.alphas is None:
        args.alphas = [args.alpha]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = load_jsonl(args.corpus)
    corpus_ids = [p["passage_id"] for p in corpus]
    corpus_texts = [p["text"] for p in corpus]

    # encode corpus once (dense side)
    if args.dense == "datr":
        assert args.datr_checkpoint, "--datr_checkpoint required for --dense datr"
        dense = DATRWrapper(args.datr_checkpoint, args.config, device, no_era=args.no_era)
        print("Encoding corpus with DATR...")
        args._corpus_emb = dense.encode_corpus(corpus, args.batch_size)
    else:
        name, pool, qpref, ppref = HF[args.dense]
        dense = HFEncoder(name, pool, device)
        print(f"Encoding corpus with {name}...")
        args._corpus_emb = dense.encode([ppref + t for t in corpus_texts],
                                        args.batch_size, args.max_passage_len)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {"dense": args.dense, "alphas": args.alphas, "cand": args.cand}

    for tag, path in [("original", args.queries), ("modern", args.queries_modern)]:
        if not path or not os.path.exists(path):
            continue
        data = load_jsonl(path)
        if args.max_queries:
            data = data[:args.max_queries]
        f = "modern_query" if (tag == "modern" and "modern_query" in data[0]) else "query"
        by_alpha = run(corpus, corpus_ids, corpus_texts, data, f, dense, args, device, tag)
        results[tag] = {str(a): m for a, m in by_alpha.items()}
        print(f"\n=== Hybrid BM25+{args.dense}  |  {tag}  (alpha = weight on BM25) ===")
        print(f"{'alpha':>6} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@100':>7} {'MRR':>7}")
        for a in args.alphas:
            m = by_alpha[a]
            print(f"{a:>6.2f} {m['recall@1']:>7.2f} {m['recall@5']:>7.2f} "
                  f"{m['recall@10']:>7.2f} {m['recall@100']:>7.2f} {m['mrr']:>7.2f}")

    if "original" in results and "modern" in results:
        print(f"\n=== Vocabulary gap (R@1 original - modern) by alpha ===")
        for a in args.alphas:
            o = results["original"][str(a)]["recall@1"]
            mo = results["modern"][str(a)]["recall@1"]
            print(f"  alpha={a:.2f}:  orig {o:.2f}  modern {mo:.2f}  gap {mo - o:+.2f}")

    out = os.path.join(args.output_dir, f"hybrid_{args.dense}_sweep.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
