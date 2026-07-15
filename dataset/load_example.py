#!/usr/bin/env python3
"""
Minimal loader for the diachronic modern queries.

Usage:
    python load_example.py modern_queries_test.jsonl [path/to/caqa_corpus.jsonl]

Shows a few records and, if a ChroniclingAmericaQA corpus file is given, joins each
question to its gold passage text via positive_passage_id.
"""
import json, sys

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    queries = load_jsonl(sys.argv[1])
    print(f"{len(queries)} queries")
    reworded = sum(q["modern_query"].strip() != q["original_question"].strip()
                   for q in queries if q.get("modern_query"))
    print(f"modern != original: {reworded} ({100*reworded/len(queries):.1f}%)")

    pid2text = {}
    if len(sys.argv) > 2:                      # optional: join to gold passages
        for p in load_jsonl(sys.argv[2]):
            pid2text[p.get("passage_id")] = p.get("text", "")

    for q in queries[:3]:
        print("\n" + "=" * 70)
        print("original:", q["original_question"])
        print("modern  :", q["modern_query"])
        print("answer  :", q["answer"], "| year:", q["year"],
              "| pid:", q["positive_passage_id"])
        if pid2text:
            print("passage :", pid2text.get(q["positive_passage_id"], "")[:200], "...")

if __name__ == "__main__":
    main()
