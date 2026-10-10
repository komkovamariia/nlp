# NLP experiments: from classical baselines to modern LLM systems

This repository is a portfolio of reproducible NLP experiments. It starts with tokenization, linear text classification and static embeddings, then develops Transformer internals, named-entity recognition, retrieval-augmented generation, parameter-efficient fine-tuning and preference optimization.

Each notebook states its evaluation contract and separates model selection from final testing. Reusable algorithms live in `src/nlp_portfolio`; unit tests cover preprocessing, BIO alignment, ranked retrieval, attention, calibration, LoRA and DPO. Expensive model downloads are explicit and optional whenever a smaller deterministic demonstration is available.

## Notebooks

| Notebook | Experiment |
| --- | --- |
| [Text preprocessing](notebooks/00_text_preprocessing.ipynb) | Tokenization, stemming, lemmatization and Zipf statistics |
| [Text classification](notebooks/01_text_classification.ipynb) | TF-IDF with logistic regression and train-only model selection |
| [Word embeddings](notebooks/02_embeddings.ipynb) | Word2Vec, retrofitting and orthogonal alignment |
| [Causal LM surprisal](notebooks/03_causal_lm_typos.ipynb) | GPT-2 token probabilities for typo detection |
| [Named-entity recognition](notebooks/04_named_entity_recognition.ipynb) | BERT and Gemma on WikiANN with entity-level F1 |
| [Retrieval QA](notebooks/05_retrieval_qa.ipynb) | Sparse retrieval and evidence-based multiple-choice answers |
| [Attention](notebooks/06_attention_mechanics.ipynb) | Dot-product attention and causal masking |
| [Retrieval metrics](notebooks/07_retrieval_metrics.ipynb) | Precision@k, reciprocal rank and nDCG |
| [Representation baselines](notebooks/08_transformer_baselines.ipynb) | TF-IDF and frozen Transformer features |
| [BPE and Transformer from scratch](notebooks/09_bpe_and_transformer_from_scratch.ipynb) | BPE merges, masks and a minimal PyTorch encoder |
| [LoRA and quantization](notebooks/10_lora_and_quantization.ipynb) | Low-rank updates, trainable parameters and memory accounting |
| [Hybrid RAG and reranking](notebooks/11_hybrid_rag_and_reranking.ipynb) | Multi-channel retrieval, RRF, reranking and failure traces |
| [Preference optimization and evaluation](notebooks/12_preference_optimization_and_llm_evaluation.ipynb) | DPO loss, judge position bias and uncertainty |
| [Biomedical evidence mining](notebooks/13_biomedical_evidence_mining.ipynb) | Gene normalization, provenance and evidence retrieval |

## Setup

Python 3.10 or newer:

```bash
python -m pip install -e '.[notebooks,test]'
python -m pytest -q
python scripts/validate_notebooks.py
```

Open a notebook from the repository root so imports from `src/nlp_portfolio` resolve after installation. Some experiments download datasets or model weights. Gemma, dense RAG and generative evaluation benefit from a GPU. Record package versions, model revisions and hardware when reporting new measurements.

## Evaluation

Classification uses held-out data after hyperparameter selection and reports calibration alongside discrimination. NER uses exact entity-level precision, recall and F1 on WikiANN; `O` tokens do not count as entities. Retrieval metrics require independent relevance labels. RAG predictions retain evidence identifiers so retrieval and generation errors can be separated.

Unit tests check the shared algorithms. End-to-end GPU experiments must be rerun to obtain current measurements; no pretrained-model scores are presented as validated results in this repository.

## Sources

The initial experiments were developed from NLP exercises at the University of Padova. The advanced sequence follows the HSE materials on Transformers, BERT and GPT, modern LLMs, PEFT, model compression, RAG and evaluation. Dataset, course and primary-method references are collected in [sources/references.md](sources/references.md). The notebooks are independent implementations; the original teaching files are not redistributed.
