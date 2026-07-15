"""
End-to-end QA reader evaluation over pre-computed retrieval runs.

This is the §6 evaluator. It decouples reading from retrieving: hybrid_eval.py
--dump_runs writes the top-k retrieved passages per query (so the QA numbers sit on
exactly the retrieval reported in Tables 1-2), and this script feeds those passages to
a reader and scores EM/F1/contains.

All readers use the SAME prompt and the SAME metric, so rows are directly comparable.
The point (reviewer MAJOR-3) is a *fair* LLM comparison:

  - closed-book LLM   : base instruct LLM, NO retrieved context  -> near chance
                        (run with --no_context)
  - instruction LLM   : base instruct LLM + retrieved context    (no --adapter_path)
  - QLoRA-FT LLM      : base instruct LLM + adapter + context     (--adapter_path ...)
  - extractive reader : BERT/RoBERTa-squad over retrieved context (--reader extractive)

Examples
--------
  # fair fine-tuned LLM row (best hybrid runs, modern queries)
  python qa_eval_runs.py --reader llm --runs outputs/v3/qa_runs/runs_modern.jsonl \
      --base_model meta-llama/Llama-3.1-8B-Instruct \
      --adapter_path outputs/qlora_llama/best_adapter \
      --ctx_k 5 --output outputs/v3/qa/llm_ft_modern.json

  # same, no fine-tuning (instruction-only baseline)
  python qa_eval_runs.py --reader llm --runs outputs/v3/qa_runs/runs_modern.jsonl \
      --base_model meta-llama/Llama-3.1-8B-Instruct \
      --ctx_k 5 --output outputs/v3/qa/llm_base_modern.json

  # closed-book (retrieval indispensable check)
  python qa_eval_runs.py --reader llm --runs outputs/v3/qa_runs/runs_modern.jsonl \
      --base_model meta-llama/Llama-3.1-8B-Instruct --no_context \
      --output outputs/v3/qa/llm_closedbook_modern.json

  # extractive reader
  python qa_eval_runs.py --reader extractive --runs outputs/v3/qa_runs/runs_modern.jsonl \
      --reader_model deepset/roberta-base-squad2 \
      --ctx_k 5 --output outputs/v3/qa/roberta_modern.json
"""

import os
import re
import json
import string
import argparse

import numpy as np
import torch
from tqdm import tqdm

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_MSG = ("You are a precise question-answering assistant. Answer the question "
              "based ONLY on the provided context.\n- Give a short, direct answer "
              "(usually 1-5 words)\n- If the answer is not in the context, say "
              "\"NOT FOUND\"\n- Do not make up information")


# --------------------------------------------------------------------------- #
# Metrics (identical to qlora_finetune.py / end_to_end_qa.py)
# --------------------------------------------------------------------------- #
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(p, g):
    return int(normalize_answer(p) == normalize_answer(g))


def f1_score(p, g):
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    if not pt or not gt:
        return float(pt == gt)
    common = set(pt) & set(gt)
    n = sum(min(pt.count(w), gt.count(w)) for w in common)
    if n == 0:
        return 0.0
    prec, rec = n / len(pt), n / len(gt)
    return 2 * prec * rec / (prec + rec)


def contains_score(p, g):
    return int(normalize_answer(g) in normalize_answer(p))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_runs(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_context(row, ctx_k, no_context):
    if no_context:
        return ""
    passages = row.get("retrieved", [])[:ctx_k]
    return "\n\n".join(p.get("text", "") for p in passages)


# --------------------------------------------------------------------------- #
# LLM reader
# --------------------------------------------------------------------------- #
def build_prompt(question, context, tokenizer):
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (short and direct):"
    if tokenizer.chat_template:
        msgs = [{"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_MSG}\n\n{user}"


def eval_llm(rows, args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True) \
        if args.four_bit else None
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"  loaded adapter: {args.adapter_path}")
    model.eval()

    em, f1, cont = [], [], []
    preds = []
    for r in tqdm(rows, desc="llm"):
        ctx = build_context(r, args.ctx_k, args.no_context)
        prompt = build_prompt(r["query"], ctx, tok)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_len).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        pred = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = pred.split("\n")[0].strip().strip('"\'')
        g = r.get("answer", "")
        em.append(em_score(pred, g)); f1.append(f1_score(pred, g)); cont.append(contains_score(pred, g))
        preds.append({"q": r["query"], "pred": pred, "gold": g})
    return em, f1, cont, preds


# --------------------------------------------------------------------------- #
# Extractive reader
# --------------------------------------------------------------------------- #
def eval_extractive(rows, args):
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
    tok = AutoTokenizer.from_pretrained(args.reader_model)
    mdl = AutoModelForQuestionAnswering.from_pretrained(args.reader_model)
    qa = pipeline("question-answering", model=mdl, tokenizer=tok,
                  device=0 if DEV.type == "cuda" else -1)
    em, f1, cont, preds = [], [], [], []
    for r in tqdm(rows, desc="extractive"):
        ctx = build_context(r, args.ctx_k, args.no_context)
        if not ctx.strip():
            pred = ""
        else:
            try:
                pred = qa(question=r["query"], context=ctx,
                          max_answer_len=30, handle_impossible_answer=True)["answer"]
            except Exception:
                pred = ""
        g = r.get("answer", "")
        em.append(em_score(pred, g)); f1.append(f1_score(pred, g)); cont.append(contains_score(pred, g))
        preds.append({"q": r["query"], "pred": pred, "gold": g})
    return em, f1, cont, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="jsonl from hybrid_eval.py --dump_runs")
    ap.add_argument("--reader", choices=["llm", "extractive"], default="llm")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter_path", default=None, help="PEFT adapter (fine-tuned LLM)")
    ap.add_argument("--reader_model", default="deepset/roberta-base-squad2",
                    help="extractive QA model")
    ap.add_argument("--ctx_k", type=int, default=5, help="# retrieved passages to concatenate")
    ap.add_argument("--no_context", action="store_true", help="closed-book (ignore retrieval)")
    ap.add_argument("--four_bit", action="store_true", default=True)
    ap.add_argument("--max_len", type=int, default=3500)
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_runs(args.runs)
    if args.max_eval:
        rows = rows[:args.max_eval]
    print(f"{len(rows)} examples | reader={args.reader} | ctx_k={args.ctx_k} | "
          f"{'CLOSED-BOOK' if args.no_context else 'with context'} | "
          f"adapter={'yes' if args.adapter_path else 'no'}")

    if args.reader == "llm":
        em, f1, cont, preds = eval_llm(rows, args)
    else:
        em, f1, cont, preds = eval_extractive(rows, args)

    metrics = {"em": float(100 * np.mean(em)), "f1": float(100 * np.mean(f1)),
               "contains": float(100 * np.mean(cont)), "n": len(rows)}
    result = {
        "config": {"runs": args.runs, "reader": args.reader, "base_model": args.base_model,
                   "adapter_path": args.adapter_path, "reader_model": args.reader_model,
                   "ctx_k": args.ctx_k, "no_context": args.no_context},
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    json.dump(preds[:100], open(args.output.replace(".json", ".samples.json"), "w"), indent=2)
    print(f"\nEM={metrics['em']:.2f}  F1={metrics['f1']:.2f}  contains={metrics['contains']:.2f}  "
          f"(n={metrics['n']})")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
