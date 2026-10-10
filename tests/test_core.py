import json
import numpy as np
import pytest
from nlp_portfolio.ner import align_word_labels, first_json_object, text_entities_to_bio
from nlp_portfolio.rag import build_sparse_index, retrieve, option_id, format_mcq
from nlp_portfolio.embeddings import cosine, align_orthogonal, retrofit
from nlp_portfolio.attention import scaled_dot_product_attention, causal_mask
from nlp_portfolio.retrieval_eval import reciprocal_rank, ndcg_at_k
from nlp_portfolio.classification import fit_text_classifier


def test_bio_subwords():
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    result = align_word_labels([None, 0, 0, 1, None], [1, 0], names, {x: i for i, x in enumerate(names)})
    assert result == [-100, 1, 2, 0, -100]


def test_chat_parsing_and_repetitions():
    output = '```json\n{"PERSON": [], "ORG": [], "LOC": ["New York"]}\n```\nText: example {"bad":true}'
    assert first_json_object(output)["LOC"] == ["New York"]
    assert text_entities_to_bio(["New", "York", "and", "New", "York"], {"LOC": ["New York"]}) == ["B-LOCATION", "I-LOCATION", "O", "B-LOCATION", "I-LOCATION"]


def test_retrieval():
    idx = build_sparse_index({"1": "The goalkeeper defends the goal", "2": "A match lasts ninety minutes"})
    assert retrieve("How long is a match?", idx, 1)[0]["id"] == "2"
    assert option_id("The answer is C.") == "C"
    assert option_id("UNKNOWN") is None
    assert option_id("A or B") is None
    assert "A." in format_mcq("Question?", [{"id":"A", "text":"yes"}], [{"id":"rule", "text":"some context", "score":0.9}])


def test_vector_methods():
    assert np.isclose(cosine([1, 0], [1, 0]), 1)
    identity = np.eye(3)
    assert np.allclose(align_orthogonal(identity, identity), identity)
    neighbors = {"a": ["b"], "b": ["a"]}
    values = retrofit({"a": [1.0, 0.0], "b": [0.0, 1.0]}, neighbors, iterations=1)
    assert np.allclose(values["a"], [0.5, 0.5])


def test_attention():
    matrix = np.eye(3)
    _, w = scaled_dot_product_attention(matrix, matrix, matrix, causal_mask(3))
    assert np.allclose(w.sum(axis=1), 1)
    assert np.allclose(np.triu(w, 1), 0)


def test_metrics():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert np.isclose(ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 2), 1)


def test_no_data_leakage_classification():
    sentences = [f"wonderful beautiful movie {i}" if i%2 else f"awful terrible movie {i}" for i in range(30)]
    labels = [i%2 for i in range(30)]
    m, report, params = fit_text_classifier(sentences, labels)
    assert 'accuracy' in report and 'classifier__C' in params


def test_preprocessing_zipf():
    from nlp_portfolio.preprocessing import simple_tokens, zipf_slope
    tokens = simple_tokens("Hello hello HELLO world world NLP! NLP")
    assert tokens.count('hello') == 3
    slope, _, ranks, freq = zipf_slope(tokens)
    assert len(ranks) == 3
    assert slope < 0
