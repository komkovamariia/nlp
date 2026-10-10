"""BIO labels and span matching for WikiANN."""
import json

TYPES = {"PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG", "LOC": "LOCATION", "LOCATION": "LOCATION"}


def align_word_labels(word_ids, word_labels, label_names, label_to_id, label_all_tokens=True):
    """Align word-level BIO indices to subword positions; -100 masks specials."""
    result = []
    previous = None
    for word_id in word_ids:
        if word_id is None:
            label = -100
        elif word_id != previous:
            label = int(word_labels[word_id])
        elif not label_all_tokens:
            label = -100
        else:
            original = label_names[int(word_labels[word_id])]
            label = label_to_id["I-" + original[2:]] if original.startswith("B-") else int(word_labels[word_id])
        result.append(label)
        previous = word_id
    return result


def first_json_object(generation):
    """Read the first valid JSON object, ignoring model chatter or later examples."""
    decoder = json.JSONDecoder()
    cleaned = str(generation).replace("```json", "").replace("```", "")
    for i, char in enumerate(cleaned):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return {}


def text_entities_to_bio(words, extracted):
    """Exact word-sequence matching, including repeated mentions, with overlap guards."""
    tags = ["O"] * len(words)
    if not isinstance(extracted, dict):
        return tags
    folded = [w.casefold() for w in words]
    candidates = []
    for raw_type, mentions in extracted.items():
        name = TYPES.get(str(raw_type).upper())
        if name is None or not isinstance(mentions, list):
            continue
        for mention in mentions:
            if not isinstance(mention, str) or not mention.strip():
                continue
            target = mention.strip().casefold().split()
            for i in range(len(words) - len(target) + 1):
                if folded[i:i + len(target)] == target:
                    candidates.append((i, i + len(target), name))
    for a, b, label in sorted(candidates, key=lambda x: (-(x[1] - x[0]), x[0])):
        if any(tags[i] != "O" for i in range(a, b)):
            continue
        tags[a] = f"B-{label}"
        for i in range(a + 1, b):
            tags[i] = f"I-{label}"
    return tags


def wikiann_gold_tags(tags, label_names):
    out = []
    for tag in tags:
        label = label_names[int(tag)]
        if label == "O":
            out.append(label)
        else:
            prefix, kind = label.split("-", 1)
            out.append(f"{prefix}-{TYPES[kind]}")
    return out


def pipeline_entities_to_dict(predictions):
    found = {"PERSON": [], "ORG": [], "LOCATION": []}
    for item in predictions:
        name = TYPES.get(item.get("entity_group", "").upper())
        value = item.get("word", "").strip()
        if name and value and value not in found[name]:
            found[name].append(value)
    return found


def predict_chat_entities(text, tokenizer, model, max_new_tokens=128):
    """Greedy zero-shot NER with an instruction-tuned causal LM."""
    import torch
    prompt = ("Extract named entities from this text. Return one JSON object with "
              "keys PERSON, ORG, LOC; each value is an array of exact substrings "
              "from the text. Use [] when no entity is found. Output JSON only.\n"
              f"Text: {text}")
    messages = [{"role": "user", "content": prompt}]
    batch = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
    batch = batch.to(model.device)
    model.eval()
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return first_json_object(decoded), decoded
