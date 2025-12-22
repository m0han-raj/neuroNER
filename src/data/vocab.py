import json
from collections import Counter

def build_vocabs(path):
    word_counter = Counter()
    char_counter = Counter()
    label_set = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for w, t in zip(obj["words"], obj["ner"]):
                word_counter[w] += 1
                for c in w:
                    char_counter[c] += 1
                label_set.add(t)

    word_vocab = {"<PAD>": 0, "<UNK>": 1}
    char_vocab = {"<PAD>": 0, "<UNK>": 1}
    label_vocab = {}

    for w in word_counter:
        word_vocab[w] = len(word_vocab)
    for c in char_counter:
        char_vocab[c] = len(char_vocab)
    for l in sorted(label_set):
        label_vocab[l] = len(label_vocab)

    return word_vocab, char_vocab, label_vocab
