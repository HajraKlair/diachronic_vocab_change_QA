# Diachronic Modern Queries for ChroniclingAmericaQA

Present-day paraphrases of the ChroniclingAmericaQA **test** questions, for studying
retrieval and QA under diachronic vocabulary change. Each item pairs an original
(period-phrased) question with a `modern_query` that preserves its meaning, answer,
and named entities, over the same gold passage.

## File

`modern_queries_test.jsonl` — 24,084 records, one JSON object per line.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | string | ChroniclingAmericaQA question id (e.g. `test_0`). |
| `original_question` | string | Original, period-phrased question. |
| `modern_query` | string | Modernized paraphrase. |
| `answer` | string | Gold answer. |
| `positive_passage_id` | int | Gold passage id in the ChroniclingAmericaQA corpus. |
| `year` | int | Source page year (1800–1920). |

## Use

Join on `positive_passage_id` against the ChroniclingAmericaQA corpus
(https://github.com/DataScienceUIBK/ChroniclingAmericaQA) to get the passage text.
See `load_example.py`.

## License

CC-BY-4.0 (see `LICENSE`). The underlying questions, answers, and passages are from
ChroniclingAmericaQA and remain under that dataset's license; please cite it as well.

## Citation

```bibtex
@inproceedings{oldstories2026,
  title     = {Old Stories, New Readers: Evaluating Retrieval and QA over
               Historical Newspapers under Diachronic Vocabulary Change},
  author    = {TBD},
  booktitle = {Proceedings of the ACM/IEEE Joint Conference on Digital Libraries (JCDL)},
  year      = {2026}
}

@inproceedings{piryani2024caqa,
  title     = {ChroniclingAmericaQA: A Large-scale Question Answering Dataset
               based on Historical American Newspaper Pages},
  author    = {Piryani, Bhawna and Mozafari, Jamshid and Jatowt, Adam},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference},
  year      = {2024}
}
```
