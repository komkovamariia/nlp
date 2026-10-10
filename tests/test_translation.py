import pytest
import torch

from nlp_portfolio.translation import (
    BOS,
    PAD,
    UNK,
    AttentionTranslator,
    TransformerTranslator,
    Vocabulary,
    controlled_parallel_corpus,
    encode_pairs,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(1031)
    torch.set_num_threads(2)


def test_vocabulary_unknown_and_round_trip():
    vocabulary = Vocabulary(["we read the report"])
    assert vocabulary.decode(vocabulary.encode("we read the report")) == "we read the report"
    assert vocabulary.encode("unseen")[1] == UNK
    assert vocabulary.encode("")[0] == BOS


@pytest.mark.parametrize("constructor", [AttentionTranslator, TransformerTranslator])
def test_source_padding_does_not_change_logits(constructor):
    source_vocabulary = Vocabulary(["we read", "they observe the cell"])
    target_vocabulary = Vocabulary(["noi leggiamo", "loro osservano la cellula"])
    data = encode_pairs([("we read", "noi leggiamo")], (source_vocabulary, target_vocabulary))
    model = constructor(len(source_vocabulary), len(target_vocabulary))
    model.eval()
    with torch.inference_mode():
        original = model(data.source, data.target[:, :-1])
        padded = model(torch.nn.functional.pad(data.source, (0, 5), value=PAD), data.target[:, :-1])
    assert torch.allclose(original, padded, atol=1e-5)


def test_transformer_future_tokens_are_hidden():
    model = TransformerTranslator(20, 20)
    model.eval()
    source = torch.tensor([[BOS, 4, 5, 2]])
    target = torch.tensor([[BOS, 6, 7, 8, 9]])
    changed = target.clone()
    changed[:, 3:] = 10
    with torch.inference_mode():
        original = model(source, target)
        altered = model(source, changed)
    assert torch.allclose(original[:, :3], altered[:, :3], atol=1e-5)


def test_attention_excludes_padding_and_generation_has_no_bos():
    model = AttentionTranslator(20, 20)
    source = torch.tensor([[BOS, 4, 2, PAD, PAD]])
    encoded, hidden, padding = model.encode(source)
    _, _, weights = model.decode_step(torch.tensor([BOS]), hidden, encoded, padding)
    assert torch.equal(weights[:, 3:], torch.zeros(1, 2))
    assert torch.allclose(weights.sum(-1), torch.ones(1))
    generated = model.generate(source, max_tokens=5)
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 5


def test_controlled_corpus_is_unique():
    pairs = controlled_parallel_corpus()
    assert len(pairs) == 192
    assert len(set(pairs)) == len(pairs)
