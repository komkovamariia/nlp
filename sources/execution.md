# Notebook execution

Notebooks 00–13 were executed on CPU with their default settings. Tables, plots and printed metrics are stored in the notebooks. The [execution report](../results/notebook_execution_report.json) records the execution time, package versions and SHA-256 of each notebook's code cells. Notebooks 14–15 retain the previously recorded translation runs described in [translation_validation.md](translation_validation.md).

| Notebooks | Data and scope |
| --- | --- |
| 00 | Course text corpus and WordNet; word frequencies, bootstrap interval and vocabulary growth |
| 01 | NLTK movie reviews; TF-IDF classification and calibration on held-out reviews |
| 02 | Repeated constructed sentences for Word2Vec, plus numerical alignment and retrofitting examples |
| 03 | GPT-2 inference and the course typo dataset; threshold selected on calibration data |
| 04 | Pretrained BERT on 64 WikiANN validation examples; no fine-tuning |
| 05 | 49 rulebook entries and 25 questions; sparse retrieval only, with no relevance labels or generated answers |
| 06–07 | Numerical attention checks and retrieval metrics on constructed rankings |
| 08 | Two 20 Newsgroups categories; TF-IDF and frozen MiniLM features on the same held-out documents |
| 09–10 | Small BPE/Transformer and LoRA examples; no pretrained language-model training or QLoRA run |
| 11 | Word/character TF-IDF fusion on five sentences and three constructed queries |
| 12 | DPO loss values and a manually constructed judge table; no automated judge call |
| 13 | Alias matching and retrieval on four constructed sentences; perfect mention matching here is a check of the example |

The optional GloVe, Gemma, dense rulebook retrieval, Qwen answer generation, Hugging Face PEFT and cross-encoder branches were not enabled. An executed cell containing a disabled branch does not imply that the model ran. The rulebook notebook reports no retrieval-quality metric because `qrels.json` is absent.

The host could not open Jupyter kernel sockets, so execution used the included in-process engine. It evaluates Python cells in order and captures text, rich tables and Matplotlib figures. This route supports these plain-Python notebooks; it is not a general replacement for Jupyter magics or widgets. `LOKY_MAX_CPU_COUNT=1` was used for classification because the host restricts process introspection. Corpus files were downloaded from their original sources into local caches; downloaded weights and datasets are not committed.

All 15 existing unit tests passed. Notebook structure and Python syntax checks passed for all 16 notebooks. These checks validate code execution and numerical invariants, not performance on new datasets. GPT-2's weak typo-detection result and BERT's modest transfer score are retained and discussed next to their outputs.
