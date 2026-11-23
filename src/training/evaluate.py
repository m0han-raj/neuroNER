from collections import defaultdict

import torch


def extract_entities(tags):
    entities = set()
    start, ent_type = None, None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if ent_type:
                entities.add((start, i - 1, ent_type))
            start = i
            ent_type = tag[2:]
        elif tag.startswith("I-") and ent_type == tag[2:]:
            continue
        else:
            if ent_type:
                entities.add((start, i - 1, ent_type))
                ent_type = None

    if ent_type:
        entities.add((start, len(tags) - 1, ent_type))

    return entities


def evaluate(model, dataloader, label_vocab, device):
    model.eval()
    id2label = {v: k for k, v in label_vocab.items()}

    gold_entities = defaultdict(set)
    pred_entities = defaultdict(set)

    with torch.no_grad():
        for batch in dataloader:
            words, chars, tags, mask = batch

            words = words.to(device)
            chars = chars.to(device)
            tags = tags.to(device)
            mask = mask.to(device)

            predictions = model(words, chars, mask=mask)

            for i, pred in enumerate(predictions):
                gold = [id2label[t.item()] for t in tags[i][: len(pred)]]
                pred = [id2label[p] for p in pred]

                gold_entities["all"] |= extract_entities(gold)
                pred_entities["all"] |= extract_entities(pred)

    tp = len(gold_entities["all"] & pred_entities["all"])
    fp = len(pred_entities["all"] - gold_entities["all"])
    fn = len(gold_entities["all"] - pred_entities["all"])

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1
