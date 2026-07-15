#!/usr/bin/env python3
"""
alpha_experiment.py  --  Establish the hybrid fusion weight alpha for BM25 + E5-FT.

Methodology (reviewer-proof):
  * Fusion is the EXACT pipeline from hybrid_eval.py -- this file IMPORTS
    DATRWrapper, dense_topn, minmax, fuse_and_score from it (no reimplementation),
    and runs a CANARY asserting our metric layer matches fuse_and_score at alpha=0.5.
  * alpha is TUNED on a held-out tuning partition and REPORTED on the disjoint
    remainder (no tuning on reported data).  Default: an era-stratified split of the
    modern eval file (which carries both original_question and modern_query per record).
    Use --tune_file to instead tune on a separate dev-modern set and report on the full eval.
  * Selection objective: maximize MEAN MRR over {original, modern} on the tuning set.
    (We also report the modern-only choice as a robustness check.)
  * Reporting: R@1/5/10/100, MRR (== MAP for single-positive), nDCG@10/100, with
    bootstrap 95% CIs, a paired significance test vs the better single retriever,
    stability of the selected alpha under dev bootstraps, and an oracle per-query
    upper bound (not deployable).

Place this file NEXT TO hybrid_eval.py in the repo root, then:

    python alpha_experiment.py \
        --config configs/config_e5.yaml --datr_checkpoint outputs_e5_ft/best_model.pt --no_era \
        --corpus data/processed/corpus.jsonl \
        --eval   data/processed/test_retrieval_modern.jsonl \
        --device cuda
"""
import os, json, argparse, math
from collections import defaultdict
import numpy as np

# ---- fusion = exactly the latest code (imported, not reimplemented) ----
from hybrid_eval import DATRWrapper, dense_topn, minmax, fuse_and_score, load_jsonl

GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
KS = [1, 5, 10, 100]
NDCG_KS = [10, 100]
BIG = 10**9  # sentinel rank for "gold not in fused pool"


def build_pq(bm_docs, bm_scores, dn_idx, dn_scores, corpus_ids, positives):
    """Per-query aligned (bm_norm, dn_norm) arrays over the fused union + gold index.
    Mirrors hybrid_eval.fuse_and_score's inner construction exactly (same minmax)."""
    pq = []
    for qi in range(len(positives)):
        bm = {corpus_ids[int(bm_docs[qi][j])]: float(bm_scores[qi][j]) for j in range(len(bm_docs[qi]))}
        dn = {corpus_ids[int(dn_idx[qi][j])]: float(dn_scores[qi][j]) for j in range(len(dn_idx[qi]))}
        bmn, dnn = minmax(bm), minmax(dn)
        pids = list(set(bmn) | set(dnn))
        b = np.array([bmn.get(p, 0.0) for p in pids], dtype=np.float64)
        d = np.array([dnn.get(p, 0.0) for p in pids], dtype=np.float64)
        pos = positives[qi]
        gi = pids.index(pos) if pos in set(pids) else -1
        pq.append((b, d, gi))
    return pq


def gold_rank(pqi, alpha):
    b, d, gi = pqi
    if gi < 0:
        return None
    fused = alpha * b + (1.0 - alpha) * d
    gf = fused[gi]
    # EXACT stable-sort rank, matching hybrid_eval.fuse_and_score (sorted desc; ties keep union order):
    # gold's 0-based position = (#strictly greater) + (#equal that appear earlier in the union)
    greater = int((fused > gf).sum())
    ties_before = int((fused[:gi] == gf).sum())
    return greater + ties_before + 1


def rank_matrix(pq):
    """[len(GRID), N] integer gold ranks (BIG if gold not in pool)."""
    M = np.full((len(GRID), len(pq)), BIG, dtype=np.int64)
    for ai, a in enumerate(GRID):
        for qi, pqi in enumerate(pq):
            r = gold_rank(pqi, a)
            if r is not None:
                M[ai, qi] = r
    return M


def agg(M, ai, idx):
    """Aggregate metrics for alpha-index ai over query indices idx."""
    r = M[ai, idx].astype(np.float64)
    inpool = r < BIG
    rr = np.where(inpool, 1.0 / np.where(inpool, r, 1.0), 0.0)
    out = {f"R@{k}": 100.0 * np.mean(r <= k) for k in KS}
    out["MRR"] = 100.0 * rr.mean()
    out["MAP"] = out["MRR"]  # identical for single-positive
    for k in NDCG_KS:
        g = np.where(r <= k, 1.0 / np.log2(np.where(r <= k, r, 1.0) + 1.0), 0.0)
        out[f"nDCG@{k}"] = 100.0 * g.mean()
    return out


def rr_vec(M, ai, idx):
    r = M[ai, idx].astype(np.float64)
    inpool = r < BIG
    return np.where(inpool, 1.0 / np.where(inpool, r, 1.0), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--datr_checkpoint", required=True)
    ap.add_argument("--no_era", action="store_true")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval", required=True, help="modern eval file (needs original_question + modern_query)")
    ap.add_argument("--tune_file", default=None,
                    help="optional separate tuning file (e.g. generated dev-modern). "
                         "If set, tune on it and report on the FULL --eval (option A).")
    ap.add_argument("--cand", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--tune_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--stab_boot", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="alpha_experiment_results.json")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    # ---------- corpus + dense encoder (once) ----------
    corpus = load_jsonl(args.corpus)
    corpus_ids = [p["passage_id"] for p in corpus]
    corpus_texts = [p["text"] for p in corpus]

    dense = DATRWrapper(args.datr_checkpoint, args.config, device, no_era=args.no_era)
    emb_cache = os.path.join(os.path.dirname(args.out) or ".", "corpus_emb_e5ft.npy")
    if os.path.exists(emb_cache):
        print(f">> loading cached corpus embeddings <- {emb_cache}")
        corpus_emb = np.load(emb_cache)
        assert corpus_emb.shape[0] == len(corpus), "cached embeddings don't match corpus size; delete the cache"
    else:
        print(">> encoding corpus with E5-FT (once; will be cached) ...")
        corpus_emb = dense.encode_corpus(corpus, args.batch_size)
        os.makedirs(os.path.dirname(emb_cache) or ".", exist_ok=True)
        np.save(emb_cache, corpus_emb)

    # ---------- BM25 index (once), identical tokenization to hybrid_eval.bm25_topn ----------
    import bm25s, Stemmer
    st = Stemmer.Stemmer("english")
    print(">> building stemmed BM25 index (once) ...")
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(corpus_texts, stemmer=st))

    def bm25_pools(queries):
        res = bm.retrieve(bm25s.tokenize(queries, stemmer=st), k=args.cand)
        return res.documents, res.scores

    def dense_pools(queries):
        q_emb = dense.encode_queries(queries, args.batch_size)
        return dense_topn(corpus_emb, q_emb, args.cand)

    def make_pq(records):
        oq = [r["original_question"] for r in records]
        mq = [r.get("modern_query", r["original_question"]) for r in records]
        pos = [r["positive_passage_id"] for r in records]
        bdo, bso = bm25_pools(oq); dio, dso = dense_pools(oq)
        bdm, bsm = bm25_pools(mq); dim, dsm = dense_pools(mq)
        pq_o = build_pq(bdo, bso, dio, dso, corpus_ids, pos)
        pq_m = build_pq(bdm, bsm, dim, dsm, corpus_ids, pos)
        canary = (bdm, bsm, dim, dsm, pos)
        return pq_o, pq_m, canary

    # ---------- load eval (+ optional separate tune file) ----------
    eval_data = load_jsonl(args.eval)
    print(f">> eval records: {len(eval_data)}  (option {'A: separate tune_file' if args.tune_file else 'B: held-out split'})")

    if args.tune_file:
        tune_data = load_jsonl(args.tune_file)
        print(f">> tune records: {len(tune_data)}")
        pq_o_t, pq_m_t, _ = make_pq(tune_data)
        pq_o_r, pq_m_r, canary = make_pq(eval_data)
        Mo_t, Mm_t = rank_matrix(pq_o_t), rank_matrix(pq_m_t)
        Mo_r, Mm_r = rank_matrix(pq_o_r), rank_matrix(pq_m_r)
        tune_idx = np.arange(len(tune_data)); rep_idx = np.arange(len(eval_data))
    else:
        pq_o, pq_m, canary = make_pq(eval_data)
        Mo, Mm = rank_matrix(pq_o), rank_matrix(pq_m)
        Mo_t = Mo_r = Mo; Mm_t = Mm_r = Mm
        # era-stratified disjoint split
        groups = defaultdict(list)
        for i, r in enumerate(eval_data):
            groups[r.get("era", "unknown")].append(i)
        tune_idx, rep_idx = [], []
        for e, ids in groups.items():
            ids = np.array(ids); rng.shuffle(ids)
            cut = int(round(len(ids) * args.tune_frac))
            tune_idx += list(ids[:cut]); rep_idx += list(ids[cut:])
        tune_idx = np.array(sorted(tune_idx)); rep_idx = np.array(sorted(rep_idx))
    print(f">> tune queries: {len(tune_idx)}   report queries: {len(rep_idx)}")

    # ---------- CANARY: our metric layer must match hybrid_eval.fuse_and_score at alpha=0.5 ----------
    bdm, bsm, dim, dsm, pos = canary
    official = fuse_and_score(bdm, bsm, dim, dsm, corpus_ids, pos, 0.5, KS)
    ai_half = GRID.index(0.5)
    mine = agg(Mm_r, ai_half, np.arange(Mm_r.shape[1]))
    diffs = [abs(mine["MRR"] - official["mrr"])] + [abs(mine[f"R@{k}"] - official[f"recall@{k}"]) for k in KS]
    print(f">> CANARY (modern, alpha=0.5): max |mine - fuse_and_score| = {max(diffs):.4f}")
    assert max(diffs) < 0.05, "CANARY FAILED: metric layer does not match hybrid_eval.fuse_and_score"

    # ---------- alpha sweep on the TUNING set ----------
    print("\n" + "=" * 78)
    print("TUNING SWEEP (mean MRR of original+modern, on the tuning set)")
    print(f"{'alpha':>6} {'MRR_orig':>9} {'MRR_mod':>9} {'mean_MRR':>9}")
    sweep = []
    for ai, a in enumerate(GRID):
        mo = agg(Mo_t, ai, tune_idx)["MRR"]; mm = agg(Mm_t, ai, tune_idx)["MRR"]
        sweep.append((a, mo, mm, 0.5 * (mo + mm)))
        print(f"{a:>6.2f} {mo:>9.2f} {mm:>9.2f} {0.5*(mo+mm):>9.2f}")
    a_star = max(sweep, key=lambda x: x[3])[0]
    a_star_modern = max(sweep, key=lambda x: x[2])[0]
    print(f"\n>> selected alpha* (mean-MRR)   = {a_star}")
    print(f">> robustness: alpha* (modern-only) = {a_star_modern}")

    # ---------- REPORT on the held-out report set ----------
    def neighbors(a):
        cand = sorted({round(max(0.0, min(1.0, a + d)), 2) for d in (-0.10, -0.05, 0.0, 0.05, 0.10)})
        return [c for c in cand if c in GRID]
    rows = [("pure E5-FT (a=0)", 0.0), ("pure BM25 (a=1)", 1.0)] + \
           [(f"alpha={a}", a) for a in neighbors(a_star)]
    print("\n" + "=" * 78)
    print("REPORT-SET METRICS (held-out)   [orig | modern]")
    hdr = "R@1   R@5   R@10  R@100 MRR   nDCG10 nDCG100"
    print(f"{'setting':<20} cond  {hdr}")
    report = {}
    for name, a in rows:
        ai = GRID.index(a)
        for cond, M in (("orig", Mo_r), ("modern", Mm_r)):
            m = agg(M, ai, rep_idx)
            report[f"{name}|{cond}"] = m
            print(f"{name:<20} {cond:<5} "
                  f"{m['R@1']:5.1f} {m['R@5']:5.1f} {m['R@10']:5.1f} {m['R@100']:5.1f} "
                  f"{m['MRR']:5.2f} {m['nDCG@10']:6.2f} {m['nDCG@100']:6.2f}")

    # ---------- bootstrap 95% CIs at alpha* (report set) ----------
    ai = GRID.index(a_star)
    def boot_ci(M, idx, B):
        v = rr_vec(M, ai, idx)
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
        return 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
    ci_o = boot_ci(Mo_r, rep_idx, args.boot); ci_m = boot_ci(Mm_r, rep_idx, args.boot)
    print(f"\n>> MRR 95% CI @ alpha*={a_star}:  orig [{ci_o[0]:.2f}, {ci_o[1]:.2f}]   modern [{ci_m[0]:.2f}, {ci_m[1]:.2f}]")

    # ---------- paired significance vs the better single retriever (modern) ----------
    rr_h = rr_vec(Mm_r, GRID.index(a_star), rep_idx)
    rr_d = rr_vec(Mm_r, GRID.index(0.0), rep_idx)
    rr_b = rr_vec(Mm_r, GRID.index(1.0), rep_idx)
    base_name, rr_base = ("pure E5-FT", rr_d) if rr_d.mean() >= rr_b.mean() else ("pure BM25", rr_b)
    diff = rr_h - rr_base
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(args.boot)])
    p_one = float((bs <= 0).mean())
    print(f">> Hybrid(alpha*) vs better single retriever ({base_name}), modern: "
          f"dMRR = {100*diff.mean():+.2f}  95% CI [{100*np.percentile(bs,2.5):+.2f}, {100*np.percentile(bs,97.5):+.2f}]  "
          f"p(one-sided) = {p_one:.4f}")

    # ---------- stability of the SELECTED alpha (bootstrap the tuning set) ----------
    RRo_t = np.vstack([rr_vec(Mo_t, ai, tune_idx) for ai in range(len(GRID))])  # [A, n_tune]
    RRm_t = np.vstack([rr_vec(Mm_t, ai, tune_idx) for ai in range(len(GRID))])
    sel = []
    n = RRo_t.shape[1]
    for _ in range(args.stab_boot):
        bi = rng.integers(0, n, n)
        mean_mrr = 0.5 * (RRo_t[:, bi].mean(1) + RRm_t[:, bi].mean(1))
        sel.append(GRID[int(mean_mrr.argmax())])
    sel = np.array(sel)
    print(f">> alpha* stability over {args.stab_boot} dev bootstraps: "
          f"median {np.median(sel):.2f}, 5-95% [{np.percentile(sel,5):.2f}, {np.percentile(sel,95):.2f}], "
          f"P(=alpha*) {100*np.mean(sel==a_star):.0f}%")

    # ---------- oracle per-query alpha (upper bound, report set, modern) ----------
    best_rank_m = Mm_r[:, rep_idx].min(axis=0).astype(np.float64)
    orr = np.where(best_rank_m < BIG, 1.0 / np.where(best_rank_m < BIG, best_rank_m, 1.0), 0.0)
    print(f">> ORACLE per-query alpha (modern, upper bound, NOT deployable): MRR = {100*orr.mean():.2f}  "
          f"(fixed alpha* modern MRR = {agg(Mm_r, GRID.index(a_star), rep_idx)['MRR']:.2f})")

    # ---------- save ----------
    json.dump({
        "alpha_star_mean": a_star, "alpha_star_modern": a_star_modern,
        "grid": GRID, "tune_sweep": sweep,
        "n_tune": int(len(tune_idx)), "n_report": int(len(rep_idx)),
        "report": report,
        "ci_orig_MRR": ci_o, "ci_modern_MRR": ci_m,
        "sig_vs": base_name, "sig_dMRR": float(100*diff.mean()), "sig_p_one_sided": p_one,
        "alpha_stability_median": float(np.median(sel)),
        "oracle_modern_MRR": float(100*orr.mean()),
        "settings": {"cand": args.cand, "tune_frac": args.tune_frac, "seed": args.seed,
                     "option": "A" if args.tune_file else "B"},
    }, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
