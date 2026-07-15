#!/usr/bin/env python3
"""
ONE-SHOT end-to-end QA. For the original and modern test questions it runs the
QLoRA Llama-3.1 reader in TWO settings and reports EM / F1 / Precision / Recall /
Contains for each:

  * hybrid@10 : BM25 + E5-FT (alpha=0.60) retrieves top-10 passages -> reader
                (the realistic open-domain number; matches tab:hybrid retrieval)
  * oracle    : the gold passage is the context -> reader
                (apples-to-apples with ChroniclingAmericaQA Tables 5/6, which feed
                 the gold paragraph; compare to LLaMA2-7B / 70B rows)

Prompt, SYSTEM_MSG and metrics are copied verbatim from qa_eval_runs.py, so numbers
are consistent with your harness. Pure inference (loads your existing adapter).

    cd ~/Documents/Work/Projects/jcdl_datr_26 && conda activate datr
    python qa_all.py \
        --config configs/config_e5.yaml --ckpt outputs_e5_ft/best_model.pt \
        --corpus  data/processed/corpus.jsonl \
        --queries data/processed/test_retrieval.jsonl \
        --queries_modern data/processed/test_retrieval_modern.jsonl \
        --no_era --device cuda            # add --max_samples 100 for a smoke test
"""
import os, re, json, string, argparse, random
import numpy as np
import torch
from tqdm import tqdm
from retrieval_cache import load_artifacts

BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"      # QLoRA base
ADAPTER_DIR = "outputs/qlora_llama/best_adapter"      # QLoRA adapter
SYSTEM_MSG  = ("You are a precise question-answering assistant. Answer the question "
               "based ONLY on the provided context.\n- Give a short, direct answer "
               "(usually 1-5 words)\n- If the answer is not in the context, say "
               "\"NOT FOUND\"\n- Do not make up information")
TOP_K, ALPHA, CAND = 10, 0.60, 1000
MAX_LEN, MAX_NEW, GEN_BATCH = 3000, 16, 4   # 24 GB-safe defaults; override via CLI

# ----- metrics (verbatim from qa_eval_runs.py, + precision/recall) -----------
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em_score(p, g): return int(normalize(p) == normalize(g))
def contains_score(p, g): return int(normalize(g) in normalize(p))

def prf(p, g):
    pt, gt = normalize(p).split(), normalize(g).split()
    if not pt or not gt:
        v = float(pt == gt); return v, v, v
    common = set(pt) & set(gt)
    n = sum(min(pt.count(w), gt.count(w)) for w in common)
    if n == 0: return 0.0, 0.0, 0.0
    prec, rec = n / len(pt), n / len(gt)
    return prec, rec, 2 * prec * rec / (prec + rec)

def score(pred, golds):
    em = max(em_score(pred, g) for g in golds)
    cont = max(contains_score(pred, g) for g in golds)
    p, r, f = max((prf(pred, g) for g in golds), key=lambda t: t[2])
    return em, p, r, f, cont

# ----- reader (chat prompt = qa_eval_runs.build_prompt) ----------------------
class Reader:
    def __init__(self, base, adapter):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
        self.model = (PeftModel.from_pretrained(m, adapter) if adapter else m).eval()

    def build(self, q, ctx):
        user = f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer (short and direct):"
        if self.tok.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        return f"{SYSTEM_MSG}\n\n{user}"

    @torch.no_grad()
    def gen(self, prompts):
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_LEN, add_special_tokens=not bool(self.tok.chat_template)
                       ).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        res = []
        for g in gen:
            t = self.tok.decode(g, skip_special_tokens=True).strip()
            t = re.sub(r"^answer\s*:\s*", "", t, flags=re.I).split("\n")[0].strip()
            res.append(t)
        del out, enc, gen
        torch.cuda.empty_cache()
        return res

# ----------------------------------------------------------------------------
def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def first(d, ks):
    for k in ks:
        if d.get(k): return d[k]
    return None

def main():
    global GEN_BATCH, MAX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True); ap.add_argument("--queries", required=True)
    ap.add_argument("--queries_modern", default=None)
    ap.add_argument("--cache_dir", default="probe_cache"); ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no_era", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--settings", nargs="+", default=["hybrid", "oracle"])
    ap.add_argument("--gen_batch", type=int, default=GEN_BATCH)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--seed", type=int, default=13, help="random subset seed for --max_samples")
    ap.add_argument("--out", default="qa_all_results.json")
    a = ap.parse_args()
    GEN_BATCH, MAX_LEN = a.gen_batch, a.max_len

    import bm25s
    bm, C, enc, texts, pid2i, corpus = load_artifacts(
        a.config, a.ckpt, a.corpus, cache_dir=a.cache_dir,
        no_era=a.no_era, device=a.device, rebuild=a.rebuild)
    C = np.ascontiguousarray(C, dtype=np.float32)
    id_by_idx = [c["passage_id"] for c in corpus]
    reader = Reader(BASE_MODEL, ADAPTER_DIR)

    base = {str(d["query_id"]): d for d in load_jsonl(a.queries) if "query_id" in d}
    modern = {str(d.get("query_id")): d for d in load_jsonl(a.queries_modern)} if a.queries_modern else {}
    items_order = list(base.items())
    if a.max_samples:                                   # seeded random subset, same qids both splits
        random.Random(a.seed).shuffle(items_order)

    def build(which):
        out = []
        for qid, d in items_order:
            gp = d.get("positive_passage_id"); gi = pid2i.get(int(gp)) if gp is not None else None
            golds = [g for g in {d.get("answer"), d.get("original_answer")} if g]
            q = (first(d, ["original_question", "query", "question"]) if which == "original"
                 else first(modern.get(qid, {}), ["modern_query", "query", "modern", "rewrite"]))
            if q and golds and gi is not None:
                out.append((q, golds, int(gp), gi))
        return out[:a.max_samples] if a.max_samples else out

    def hybrid_topk(items):
        topk = []
        for s in range(0, len(items), a.bs):
            batch = items[s:s+a.bs]
            qv = enc.encode_queries([q for q, _, _, _ in batch])
            qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
            D = C @ qv.T
            for j, (q, _, _, _) in enumerate(batch):
                d_s = D[:, j]
                bm_s = bm.get_scores(bm25s.tokenize([q], stopwords="en",
                                     return_ids=False, show_progress=False)[0])
                bt = np.argpartition(bm_s, -CAND)[-CAND:]; dt = np.argpartition(d_s, -CAND)[-CAND:]
                bmn = {int(i): (bm_s[i]-bm_s[bt].min())/(bm_s[bt].max()-bm_s[bt].min()+1e-9) for i in bt}
                dnn = {int(i): (d_s[i]-d_s[dt].min())/(d_s[dt].max()-d_s[dt].min()+1e-9) for i in dt}
                fused = {i: ALPHA*bmn.get(i, 0.0)+(1-ALPHA)*dnn.get(i, 0.0) for i in set(bmn)|set(dnn)}
                topk.append(sorted(fused, key=fused.get, reverse=True)[:TOP_K])
            print(f"   retrieve {min(s+a.bs,len(items))}/{len(items)}", end="\r")
        print(); return topk

    def read_eval(items, contexts):
        EM=P=R=F=CT=0.0; preds=[]
        order = sorted(range(len(items)), key=lambda i: len(contexts[i]))  # uniform batches
        for s in tqdm(range(0, len(order), GEN_BATCH), desc="read"):
            idx = order[s:s+GEN_BATCH]
            prompts = [reader.build(items[i][0], contexts[i]) for i in idx]
            ans = reader.gen(prompts)
            for i, pred in zip(idx, ans):
                q, golds = items[i][0], items[i][1]
                em, p, r, f, ct = score(pred, golds)
                EM+=em; P+=p; R+=r; F+=f; CT+=ct
                preds.append({"q": q, "pred": pred, "gold": golds})
        n=len(items)
        return {"EM":100*EM/n, "F1":100*F/n, "P":100*P/n, "R":100*R/n,
                "Contains":100*CT/n, "n":n, "preds":preds}

    results = {}
    for split in (["original"] + (["modern"] if a.queries_modern else [])):
        items = build(split)
        print(f"\n================  {split.upper()}  (n={len(items)})  ================")
        tk = hybrid_topk(items) if "hybrid" in a.settings else None
        if tk is not None:
            r1 = 100*np.mean([items[i][2]==id_by_idx[tk[i][0]] for i in range(len(items))])
            rk = 100*np.mean([items[i][2] in [id_by_idx[x] for x in tk[i]] for i in range(len(items))])
            print(f"retrieval: R@1 {r1:.1f}  R@{TOP_K} {rk:.1f}")
        for setting in a.settings:
            if setting == "hybrid":
                ctx = [" \n\n".join(texts[i] for i in tk[j]) for j in range(len(items))]
            else:  # oracle gold passage
                ctx = [texts[gi] for (_, _, _, gi) in items]
            m = read_eval(items, ctx)
            print(f"[{split}/{setting:6}]  EM {m['EM']:.1f}  F1 {m['F1']:.1f}  "
                  f"P {m['P']:.1f}  R {m['R']:.1f}  Contains {m['Contains']:.1f}")
            results[f"{split}/{setting}"] = {k: v for k, v in m.items() if k != "preds"}
            results[f"{split}/{setting}/preds"] = m["preds"]
    json.dump(results, open(a.out, "w"), indent=2)
    print("\nsaved", a.out)

if __name__ == "__main__":
    main()
