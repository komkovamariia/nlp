# NLP exercises and experiments

Course-based experiments in text classification, embeddings, attention, retrieval and translation. Some notebooks use public datasets and pretrained models; others work through a method on a small constructed example.

The classification and translation experiments separate model selection from final testing. Reusable algorithms live in `src/nlp_portfolio`; unit tests cover preprocessing, BIO alignment, ranked retrieval, attention, calibration, LoRA and DPO. Expensive model downloads are explicit and optional whenever a smaller deterministic demonstration is available.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| [Text preprocessing](notebooks/00_text_preprocessing.ipynb) | Tokenization, stemming, lemmatization and Zipf statistics |
| [Text classification](notebooks/01_text_classification.ipynb) | TF-IDF with logistic regression and train-only model selection |
| [Word embeddings](notebooks/02_embeddings.ipynb) | Word2Vec, retrofitting and orthogonal alignment |
| [Causal LM surprisal](notebooks/03_causal_lm_typos.ipynb) | GPT-2 token probabilities for typo detection |
| [Named-entity recognition](notebooks/04_named_entity_recognition.ipynb) | BERT on WikiANN with entity-level F1; optional Gemma comparison |
| [Retrieval QA](notebooks/05_retrieval_qa.ipynb) | Sparse retrieval and evidence-based multiple-choice answers |
| [Attention](notebooks/06_attention_mechanics.ipynb) | Dot-product attention and causal masking |
| [Retrieval metrics](notebooks/07_retrieval_metrics.ipynb) | Precision@k, reciprocal rank and nDCG |
| [Representation baselines](notebooks/08_transformer_baselines.ipynb) | TF-IDF and frozen Transformer features |
| [BPE and Transformer from scratch](notebooks/09_bpe_and_transformer_from_scratch.ipynb) | BPE merges, masks and a minimal PyTorch encoder |
| [LoRA and quantization](notebooks/10_lora_and_quantization.ipynb) | Low-rank updates, trainable parameters and memory accounting |
| [Hybrid RAG and reranking](notebooks/11_hybrid_rag_and_reranking.ipynb) | Multi-channel retrieval, RRF, reranking and failure traces |
| [Preference optimization and evaluation](notebooks/12_preference_optimization_and_llm_evaluation.ipynb) | DPO loss, judge position bias and uncertainty |
| [Gene mentions and sentence retrieval](notebooks/13_biomedical_evidence_mining.ipynb) | Alias matching and TF-IDF retrieval on four example sentences |
| [Recurrent translation](notebooks/14_seq2seq_translation.ipynb) | Packed GRU, additive attention, teacher forcing and greedy generation |
| [Encoder-decoder Transformer](notebooks/15_encoder_decoder_transformer.ipynb) | Source padding, causal target masks and autoregressive translation |

## Setup

Python 3.10 or newer:

```bash
python -m pip install -e '.[notebooks,test]'
python -m pytest -q
python scripts/validate_notebooks.py
```

Open a notebook from the repository root so imports from `src/nlp_portfolio` resolve after installation. Some experiments download datasets or model weights. Gemma, dense RAG and generative evaluation benefit from a GPU. Record package versions, model revisions and hardware when reporting new measurements.

## Saved outputs

Notebooks 00–13 include outputs from their default CPU runs. The two translation notebooks retain their earlier full-run outputs. See [execution notes](sources/execution.md) for datasets, optional branches and limits, and [the run report](results/notebook_execution_report.json) for package versions and source hashes.

To rerun notebooks 00–13 from the repository root:

```bash
PYTHONHASHSEED=42 LOKY_MAX_CPU_COUNT=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/execute_notebooks.py --inprocess notebooks/0*.ipynb notebooks/1[0-3]*.ipynb
```

Omit `--inprocess` to use Jupyter. The runner saves a notebook after all its cells succeed and records failures separately. Download access is needed for the public corpora and model checkpoints on the first run.

## Evaluation

Classification uses held-out data after hyperparameter selection and reports calibration alongside discrimination. NER uses exact entity-level precision, recall and F1 on WikiANN; `O` tokens do not count as entities. Retrieval metrics require independent relevance labels. RAG predictions retain evidence identifiers so retrieval and generation errors can be separated.

Unit tests check the shared algorithms. Saved outputs record the default notebook paths, including CPU inference with GPT-2 and BERT. Optional generation, Gemma, pretrained PEFT and cross-encoder branches remain disabled unless explicitly enabled; outputs from the default run do not validate those branches.

## Sources

The initial experiments were developed from NLP exercises at the University of Padova. The advanced sequence follows the HSE materials on Transformers, BERT and GPT, modern LLMs, PEFT, model compression, RAG and evaluation. Dataset, course and primary-method references are collected in [sources/references.md](sources/references.md). The notebooks are independent implementations; the original teaching files are not redistributed.

## Translation projects

The two translation notebooks reimplement ML4HD's sequence models in PyTorch. Their default English-Italian grammar corpus is synthetic and small. Recorded scores measure held-out sentence combinations; natural translation quality requires a real parallel corpus. Set `PARALLEL_TSV` to a two-column UTF-8 parallel file to use the same pipeline with your data.

Run these notebooks with `python scripts/execute_translation.py`. Causal-mask and source-padding invariants are checked during execution and in `tests/test_translation.py`. Generated examples and execution metadata are saved in `results`; figures are saved in `figures`.
