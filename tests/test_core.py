import json
import numpy as np
import pytest
from nlp_portfolio.ner import align_word_labels, first_json_object, text_entities_to_bio
from nlp_portfolio.rag import build_sparse_index, retrieve, option_id, format_mcq
from nlp_portfolio.embeddings import cosine, align_orthogonal, retrofit
from nlp_portfolio.attention import (scaled_dot_product_attention, causal_mask,
                                     split_heads, merge_heads, rotary_embedding)
from nlp_portfolio.retrieval_eval import (reciprocal_rank, ndcg_at_k, recall_at_k,
                                          average_precision, evaluate_run)
from nlp_portfolio.classification import fit_text_classifier
from nlp_portfolio.rag import reciprocal_rank_fusion
from nlp_portfolio.evaluation import expected_calibration_error, bootstrap_interval
from nlp_portfolio.lora import lora_parameter_count, lora_linear, model_memory_gib
from nlp_portfolio.preference import dpo_loss


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
    x = np.arange(24).reshape(3, 8)
    assert np.array_equal(merge_heads(split_heads(x, 2)), x)
    assert rotary_embedding(np.ones((3, 4))).shape == (3, 4)


def test_metrics():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert np.isclose(ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 2), 1)
    assert recall_at_k(["a", "b"], {"b", "c"}, 2) == 0.5
    assert average_precision(["x", "b", "c"], {"b", "c"}) == pytest.approx((0.5 + 2/3) / 2)
    report = evaluate_run({"q": ["a", "b"]}, {"q": {"b"}}, k=2)
    assert report["rr"] == 0.5


def test_advanced_helpers():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0] in {"a", "b"}
    assert expected_calibration_error([0, 1], [0.1, 0.9], bins=2) == pytest.approx(0.1)
    low, high = bootstrap_interval([1, 2, 3], samples=100, seed=1)
    assert low <= 2 <= high
    assert lora_parameter_count(8, 4, 2) == 24
    x, w = np.ones((2, 3)), np.ones((4, 3))
    assert lora_linear(x, w, np.ones((3, 2)), np.ones((2, 4))).shape == (2, 4)
    assert model_memory_gib(1024**3, 8) == pytest.approx(1.0)
    assert dpo_loss([2.0], [0.0], [1.0], [0.0])[0] > 0


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
