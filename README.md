# NLP

Experiments in text classification, word representations, named-entity recognition and retrieval-augmented question answering.

The notebooks use standard datasets and small pretrained models. Each experiment can be run separately. The Python modules contain reusable implementations; the tests cover preprocessing, BIO alignment, retrieval and numerical routines.

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

## Setup

Python 3.10 or newer:

```bash
python -m pip install -e '.[notebooks,test]'
python -m pytest -q
python scripts/validate_notebooks.py
```

Open a notebook from the repository root so imports from `src/nlp_portfolio` resolve after installation. Some experiments download datasets or model weights. Gemma and the RAG generator may need a GPU; the public model checkpoints can change, so record the resolved version when reproducing results.

## Evaluation

Classification uses held-out data after hyperparameter selection. NER uses exact entity-level precision, recall and F1 on WikiANN; `O` tokens do not count as entities. Retrieval metrics require independent relevance labels. The generative NER experiment uses Gemma, not Llama.

Unit tests check the shared algorithms. End-to-end GPU experiments must be rerun to obtain current measurements; no pretrained-model scores are presented as validated results in this repository.

## Sources

The initial experiments were developed from NLP exercises at the University of Padova. The supplementary attention and model-comparison notebooks draw on topics covered by the [HSE NLP course](https://github.com/ashaba1in/hse-nlp/tree/2026). Dataset and library references are collected in [sources/references.md](sources/references.md). The notebooks are independent implementations; the original teaching files are not redistributed.
