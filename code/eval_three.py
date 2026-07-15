#!/usr/bin/env python3
"""
Definitive head-to-head: BM25 vs E5-FT vs Hybrid(alpha=0.60) on the TEST set.
Reports R@1, R@10, R@100, nDCG@10, MRR for original AND modern phrasings, all
from one consistent pipeline. Single gold passage per query.

  * BM25   : rank gold by full-corpus BM25 scores.
  * E5-FT  : rank gold by full-corpus dense (cosine) scores.
  * Hybrid : CANONICAL fusion -- top-`cand` BM25 pool + top-`cand` dense pool,
             per-pool min-max, union, score = a*BM25n + (1-a)*E5FTn  (a=0.60).
             (identical to hybrid_eval.py's fuse_and_score.)

Uses the same cached artifacts as the probes (retrieval_cache.load_artifacts).

    cd ~/Documents/Work/Projects/jcdl_datr_26
    conda activate datr
    python eval_three.py \
        --config configs/config_e5.yaml --ckpt outputs_e5_ft/best_model.pt \
        --corpus  data/processed/corpus.jsonl \
        --queries data/processed/test_retrieval.jsonl \
        --queries_modern data/processed/test_retrieval_modern.jsonl \
        --alpha 0.60 --cand 1000 --no_era --device cuda
"""
import json, argparse, math
import numpy as np
from retrieval_cache import load_artifacts

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def get_text(d, keys):
    for k in keys:
        if d.get(k): return d[k]
    return None

BIG = 10**9
def metrics_from_ranks(ranks):
    n = len(ranks)
    r1   = 100.0 * sum(r <= 1   for r in ranks) / n
    r10  = 100.0 * sum(r <= 10  for r in ranks) / n
    r100 = 100.0 * sum(r <= 100 for r in ranks) / n
    mrr  = 100.0 * sum(1.0 / r for r in ranks) / n
    ndcg = 100.0 * sum((1.0 / math.log2(r + 1)) if r <= 10 else 0.0 for r in ranks) / n
    return r1, r10, r100, ndcg, mrr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--queries_modern", default=None)
    ap.add_argument("--cache_dir", default="probe_cache"); ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.60); ap.add_argument("--cand", type=int, default=1000)
    ap.add_argument("--no_era", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--max_queries", type=int, default=0)
    a = ap.parse_args()

    import bm25s
    bm, C, enc, texts, pid2i, corpus = load_artifacts(
        a.config, a.ckpt, a.corpus, cache_dir=a.cache_dir,
        no_era=a.no_era, device=a.device, rebuild=a.rebuild)
    C = np.ascontiguousarray(C, dtype=np.float32)
    cand = a.cand; alpha = a.alpha

    base = {str(d["query_id"]): d for d in load_jsonl(a.queries) if "query_id" in d}
    modern = {}
    if a.queries_modern:
        for d in load_jsonl(a.queries_modern):
            modern[str(d.get("query_id"))] = d

    def build(which):
        out = []
        for qid, d in base.items():
            gold = d.get("positive_passage_id")
            if which == "original":
                t = get_text(d, ["original_question", "query", "question"])
            else:
                md = modern.get(qid)
                if not md: continue
                t = get_text(md, ["modern_query", "query", "modern", "rewrite"])
            gi = pid2i.get(int(gold)) if gold is not None else None
            if t and gi is not None:
                out.append((t, gi))
        if a.max_queries: out = out[:a.max_queries]
        return out

    def evaluate(items):
        rk = {"BM25": [], "E5-FT": [], "Hybrid": []}
        for s in range(0, len(items), a.bs):
            batch = items[s:s + a.bs]
            qv = enc.encode_queries([t for t, _ in batch])
            qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
            D = C @ qv.T                                          # (Ncorpus, B)
            for j, (t, gi) in enumerate(batch):
                d_s = D[:, j]
                bm_s = bm.get_scores(bm25s.tokenize([t], stopwords="en",
                                     return_ids=False, show_progress=False)[0])
                # full-corpus ranks for the single retrievers
                rk["BM25"].append(int((bm_s > bm_s[gi]).sum()) + 1)
                rk["E5-FT"].append(int((d_s > d_s[gi]).sum()) + 1)
                # canonical hybrid: top-cand pools, per-pool min-max, union fusion
                bt = np.argpartition(bm_s, -cand)[-cand:]
                dt = np.argpartition(d_s, -cand)[-cand:]
                bmin, bmax = bm_s[bt].min(), bm_s[bt].max()
                dmin, dmax = d_s[dt].min(), d_s[dt].max()
                bmn = {int(i): (bm_s[i] - bmin) / (bmax - bmin + 1e-9) for i in bt}
                dnn = {int(i): (d_s[i] - dmin) / (dmax - dmin + 1e-9) for i in dt}
                fused = {i: alpha * bmn.get(i, 0.0) + (1 - alpha) * dnn.get(i, 0.0)
                         for i in set(bmn) | set(dnn)}
                if gi in fused:
                    gs = fused[gi]
                    rk["Hybrid"].append(1 + sum(v > gs for v in fused.values()))
                else:
                    rk["Hybrid"].append(BIG)
            print(f"   {min(s+a.bs,len(items))}/{len(items)}", end="\r")
        print()
        return rk

    def report(tag, items):
        print(f"\n==================  {tag}  (n={len(items)})  ==================")
        rk = evaluate(items)
        print(f"{'system':<8} {'R@1':>7} {'R@10':>7} {'R@100':>7} {'nDCG@10':>8} {'MRR':>7}")
        print("-" * 50)
        for sysname in ["BM25", "E5-FT", "Hybrid"]:
            r1, r10, r100, ndcg, mrr = metrics_from_ranks(rk[sysname])
            star = "  <- alpha=%.2f" % alpha if sysname == "Hybrid" else ""
            print(f"{sysname:<8} {r1:>7.1f} {r10:>7.1f} {r100:>7.1f} {ndcg:>8.1f} {mrr:>7.1f}{star}")

    report("ORIGINAL", build("original"))
    if a.queries_modern:
        report("MODERN", build("modern"))
    print("\n(all numbers x100; ranks use strict-greater tie-breaking, single gold per query)")

if __name__ == "__main__":
    main()
