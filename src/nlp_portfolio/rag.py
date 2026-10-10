"""TF-IDF retrieval for multiple-choice question answering."""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_sparse_index(rules):
    if not isinstance(rules, dict) or not rules:
        raise ValueError("Expected a nonempty mapping of rule ids to text")
    ids = list(rules)
    texts = [str(rules[i]) for i in ids]
    vect = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    matrix = vect.fit_transform(texts)
    return ids, texts, vect, matrix


def retrieve(query, index, top_k=5):
    ids, texts, vectorizer, matrix = index
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    return [
        {"id": ids[i], "text": texts[i], "score": float(scores[i])}
        for i in np.argsort(-scores)[: max(0, top_k)]
    ]


def option_id(text):
    """Return A-E only if one unambiguous answer exists; otherwise abstain."""
    s = str(text).strip().upper()
    matches = re.findall(r"(?<![A-Z])[A-E](?![A-Z])", s)
    return matches[0] if len(matches) == 1 else None


def format_mcq(question, options, passages):
    options_text = "\n".join(f"{opt['id']}. {opt['text']}" for opt in options)
    context = "\n\n".join(f"[{item['id']}] {item['text']}" for item in passages)
    return (
        "Answer the multiple-choice question using ONLY the rule passages. "
        "If the evidence is insufficient, respond UNKNOWN. "
        "Otherwise return exactly one letter A, B, C, D, or E.\n\n"
        f"RULES:\n{context}\n\nQUESTION:\n{question}\n\nOPTIONS:\n{options_text}"
    )


def evaluate_retrieval(retrieved_ids, relevant_ids, k=5):
    """Recall@k for questions with explicit relevance annotations."""
    relevant = set(relevant_ids)
    if not relevant:
        return None
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse ranked identifier lists without requiring comparable score scales."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], str(item)))
