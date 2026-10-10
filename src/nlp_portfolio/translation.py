from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn


PAD, BOS, EOS, UNK = 0, 1, 2, 3


class Vocabulary:
    def __init__(self, sentences: list[str]):
        words = sorted({word for sentence in sentences for word in sentence.lower().split()})
        self.tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + words
        self.indices = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, sentence: str) -> list[int]:
        return [BOS] + [self.indices.get(word, UNK) for word in sentence.lower().split()] + [EOS]

    def decode(self, sequence: list[int]) -> str:
        words = []
        for index in sequence:
            if index == EOS:
                break
            if index not in (PAD, BOS):
                words.append(self.tokens[index])
        return " ".join(words)


def controlled_parallel_corpus() -> list[tuple[str, str]]:
    subjects = [
        ("i", "io", "read", "leggo", "observe", "osservo"),
        ("you", "tu", "read", "leggi", "observe", "osservi"),
        ("we", "noi", "read", "leggiamo", "observe", "osserviamo"),
        ("they", "loro", "read", "leggono", "observe", "osservano"),
    ]
    readable = [
        ("the report", "il rapporto"),
        ("the article", "l'articolo"),
        ("the book", "il libro"),
        ("the protocol", "il protocollo"),
        ("the document", "il documento"),
        ("the result", "il risultato"),
    ]
    observable = [
        ("the cell", "la cellula"),
        ("the sample", "il campione"),
        ("the image", "l'immagine"),
        ("the signal", "il segnale"),
        ("the tissue", "il tessuto"),
        ("the experiment", "l'esperimento"),
    ]
    modifiers = [
        ("today", "oggi"),
        ("carefully", "attentamente"),
        ("again", "di nuovo"),
        ("in the laboratory", "in laboratorio"),
    ]
    pairs = []
    for english, italian, read, translated_read, observe, translated_observe in subjects:
        for verb, translated_verb, objects in [
            (read, translated_read, readable),
            (observe, translated_observe, observable),
        ]:
            for object_name, translated_object in objects:
                for modifier, translated_modifier in modifiers:
                    pairs.append(
                        (
                            f"{english} {verb} {object_name} {modifier}",
                            f"{italian} {translated_verb} {translated_object} {translated_modifier}",
                        )
                    )
    return pairs


def read_parallel_tsv(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 2 or not all(column.strip() for column in columns):
            raise ValueError(f"Line {number} must contain source and target separated by one tab")
        pairs.append(tuple(column.strip().lower() for column in columns))
    return list(dict.fromkeys(pairs))


@dataclass
class ParallelData:
    source: torch.Tensor
    target: torch.Tensor


def encode_pairs(pairs: list[tuple[str, str]], vocabularies: tuple[Vocabulary, Vocabulary]) -> ParallelData:
    source_vocabulary, target_vocabulary = vocabularies
    source = [torch.tensor(source_vocabulary.encode(pair[0])) for pair in pairs]
    target = [torch.tensor(target_vocabulary.encode(pair[1])) for pair in pairs]
    return ParallelData(
        nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=PAD),
        nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=PAD),
    )


class AdditiveAttention(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.keys = nn.Linear(hidden, hidden, bias=False)
        self.queries = nn.Linear(hidden, hidden, bias=False)
        self.scores = nn.Linear(hidden, 1, bias=False)

    def forward(self, query, encoded, padding):
        scores = self.scores(torch.tanh(self.keys(encoded) + self.queries(query)[:, None])).squeeze(-1)
        weights = scores.masked_fill(padding, float("-inf")).softmax(-1)
        context = torch.bmm(weights[:, None], encoded).squeeze(1)
        return context, weights


class AttentionTranslator(nn.Module):
    def __init__(self, source_size, target_size, hidden=48):
        super().__init__()
        self.source_embedding = nn.Embedding(source_size, hidden, padding_idx=PAD)
        self.target_embedding = nn.Embedding(target_size, hidden, padding_idx=PAD)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.attention = AdditiveAttention(hidden)
        self.decoder = nn.GRU(2 * hidden, hidden, batch_first=True)
        self.output = nn.Linear(2 * hidden, target_size)

    def encode(self, source):
        lengths = source.ne(PAD).sum(-1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.source_embedding(source), lengths, batch_first=True, enforce_sorted=False
        )
        encoded, hidden = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=source.shape[1])
        return encoded, hidden, source.eq(PAD)

    def decode_step(self, token, hidden, encoded, padding):
        context, attention = self.attention(hidden[-1], encoded, padding)
        decoder_input = torch.cat([self.target_embedding(token), context], dim=-1)[:, None]
        output, hidden = self.decoder(decoder_input, hidden)
        logits = self.output(torch.cat([output[:, 0], context], dim=-1))
        return logits, hidden, attention

    def forward(self, source, target_input):
        encoded, hidden, padding = self.encode(source)
        predictions = []
        for token in target_input.unbind(1):
            logits, hidden, _ = self.decode_step(token, hidden, encoded, padding)
            predictions.append(logits)
        return torch.stack(predictions, dim=1)

    @torch.inference_mode()
    def generate(self, source, max_tokens=20):
        self.eval()
        encoded, hidden, padding = self.encode(source)
        token = torch.full((len(source),), BOS, device=source.device)
        finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
        generated = []
        for _ in range(max_tokens):
            logits, hidden, _ = self.decode_step(token, hidden, encoded, padding)
            token = logits.argmax(-1)
            token = torch.where(finished, torch.full_like(token, PAD), token)
            generated.append(token)
            finished |= token.eq(EOS)
            if finished.all():
                break
        return torch.stack(generated, dim=1)


class SinusoidalPositions(nn.Module):
    def __init__(self, dimensions=48, max_tokens=256):
        super().__init__()
        position = torch.arange(max_tokens)[:, None]
        frequency = torch.exp(torch.arange(0, dimensions, 2) * (-math.log(10000) / dimensions))
        table = torch.zeros(max_tokens, dimensions)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("table", table[None])

    def forward(self, embedded):
        if embedded.shape[1] > self.table.shape[1]:
            raise ValueError("Sequence exceeds the positional table length")
        return embedded + self.table[:, : embedded.shape[1]]


class TransformerTranslator(nn.Module):
    def __init__(self, source_size, target_size, dimensions=48):
        super().__init__()
        self.source_embedding = nn.Embedding(source_size, dimensions, padding_idx=PAD)
        self.target_embedding = nn.Embedding(target_size, dimensions, padding_idx=PAD)
        self.positions = SinusoidalPositions(dimensions)
        self.transformer = nn.Transformer(
            d_model=dimensions,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
        )
        self.output = nn.Linear(dimensions, target_size)
        self.scale = math.sqrt(dimensions)

    def forward(self, source, target_input):
        source_vectors = self.positions(self.source_embedding(source) * self.scale)
        target_vectors = self.positions(self.target_embedding(target_input) * self.scale)
        causal = torch.ones(
            target_input.shape[1], target_input.shape[1], dtype=torch.bool, device=source.device
        ).triu(1)
        hidden = self.transformer(
            source_vectors,
            target_vectors,
            tgt_mask=causal,
            src_key_padding_mask=source.eq(PAD),
            tgt_key_padding_mask=target_input.eq(PAD),
            memory_key_padding_mask=source.eq(PAD),
        )
        return self.output(hidden)

    @torch.inference_mode()
    def generate(self, source, max_tokens=20):
        self.eval()
        sequence = torch.full((len(source), 1), BOS, device=source.device)
        finished = torch.zeros(len(source), dtype=torch.bool, device=source.device)
        for _ in range(max_tokens):
            token = self(source, sequence)[:, -1].argmax(-1)
            token = torch.where(finished, torch.full_like(token, PAD), token)
            sequence = torch.cat([sequence, token[:, None]], dim=1)
            finished |= token.eq(EOS)
            if finished.all():
                break
        return sequence[:, 1:]
