import json

import torch
from torch.utils.data import Dataset


class TamilNERDataset(Dataset):
    def __init__(
        self, path, word_vocab, char_vocab, label_vocab, max_len=80, max_samples=None
    ):
        self.samples = []
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.max_len = max_len

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                words = obj["words"][:max_len]
                labels = obj["ner"][:max_len]
                self.samples.append((words, labels))

                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, labels = self.samples[idx]

        word_ids, char_ids, tag_ids = [], [], []

        for w, t in zip(words, labels):
            word_ids.append(self.word_vocab.get(w, 1))
            char_ids.append([self.char_vocab.get(c, 1) for c in w[:20]])
            tag_ids.append(self.label_vocab[t])

        return word_ids, char_ids, tag_ids
