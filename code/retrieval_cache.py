#!/usr/bin/env python3
"""
SHARED retrieval-artifact cache used by every probe_*.py script.

The two expensive artifacts -- the BM25 index and the E5-FT corpus embeddings -- are built
ONCE, saved to cache_dir (default ./probe_cache), and reloaded from disk on every subsequent
run by ANY script that imports load_artifacts().  So ">> encoding corpus with E5-FT" happens a
single time, ever, until the checkpoint or corpus changes (auto-detected via meta.json).

    from retrieval_cache import load_artifacts
    bm, C, enc, texts, pid2i, corpus = load_artifacts(config, ckpt, corpus_path,
                                                       cache_dir="probe_cache",
                                                       no_era=True, device="cuda")
"""
import os, json, numpy as np

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ---- E5-FT (DATR) encoder, identical across all probe scripts ----
class DATREnc:
    def __init__(self, config_path, ckpt, device, no_era=True):
        import torch
        from transformers import AutoTokenizer
        from models.datr import DATR
        from utils.helpers import load_config
        self.torch, self.device, self.no_era = torch, device, no_era
        cfg = load_config(config_path); m = cfg["model"]
        bins = cfg["eras"]["bins"]; self.era_to_id = {e["name"]: i for i, e in enumerate(bins)}
        self.era_to_id["unknown"] = len(bins)
        self.tok = AutoTokenizer.from_pretrained(m["encoder_name"])
        self.model = DATR(encoder_name=m["encoder_name"], era_embedding_dim=m["era_embedding_dim"],
                          num_eras=len(self.era_to_id) - 1, pooling=m["pooling"],
                          normalize=m["normalize_embeddings"], shared_encoder=False)
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        sd = raw if all(torch.is_tensor(v) for v in raw.values()) else raw.get("model_state_dict", raw.get("state_dict", raw))
        self.model.load_state_dict(sd, strict=False); self.model.to(device).eval()
    def _enc(self, texts, kind, bs, maxlen):
        import torch
        fn = self.model.encode_queries if kind == "query" else self.model.encode_passages
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i+bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out.append(fn(enc["input_ids"], enc["attention_mask"], None).cpu().numpy())
        return np.concatenate(out, 0)
    def encode_corpus(self, texts, bs=128): return self._enc(texts, "passage", bs, 256)
    def encode_queries(self, texts, bs=64):  return self._enc(texts, "query", bs, 128)

def load_artifacts(config, ckpt, corpus_path, cache_dir="probe_cache",
                   no_era=True, device="cuda", rebuild=False):
    """Return (bm, C, enc, texts, pid2i, corpus), building+saving the cache on first use."""
    import bm25s
    os.makedirs(cache_dir, exist_ok=True)
    emb_path  = os.path.join(cache_dir, "e5ft_corpus.npy")
    bm_dir    = os.path.join(cache_dir, "bm25_index")
    meta_path = os.path.join(cache_dir, "meta.json")

    corpus = load_jsonl(corpus_path)
    texts  = [c.get("text", "") for c in corpus]
    pid2i  = {int(c["passage_id"]): i for i, c in enumerate(corpus)}
    n = len(corpus)

    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    ok = (not rebuild) and meta.get("n") == n and meta.get("ckpt") == ckpt

    # ---- BM25 index (cached) ----
    if ok and os.path.isdir(bm_dir):
        print(">> loading cached BM25 index ..."); bm = bm25s.BM25.load(bm_dir, load_corpus=False)
    else:
        print(">> building BM25 index (one time) ..."); bm = bm25s.BM25()
        bm.index(bm25s.tokenize(texts, stopwords="en", show_progress=False)); bm.save(bm_dir)

    # ---- E5-FT encoder (always loaded; cheap) + corpus embeddings (cached; the slow part) ----
    enc = DATREnc(config, ckpt, device, no_era=no_era)
    if ok and os.path.exists(emb_path):
        print(">> loading cached E5-FT corpus embeddings ..."); C = np.load(emb_path)
    else:
        print(">> encoding corpus with E5-FT (one time; will be cached) ...")
        C = enc.encode_corpus(texts); C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        np.save(emb_path, C)
        json.dump({"n": n, "ckpt": ckpt, "dim": int(C.shape[1])}, open(meta_path, "w"))
        print(f">> cache written to {cache_dir}/ (reused automatically next time)")

    return bm, C, enc, texts, pid2i, corpus
