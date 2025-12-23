import torch.nn as nn
import torch
from TorchCRF import CRF
from .char_cnn import CharCNN

class NERModel(nn.Module):
    def __init__(self, word_vocab, char_vocab, label_vocab):
        super().__init__()
        self.word_emb = nn.Embedding(len(word_vocab), 100)
        self.char_enc = CharCNN(len(char_vocab))

        self.lstm = nn.LSTM(
            input_size=150,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(256, len(label_vocab))
        self.crf = CRF(len(label_vocab))

    def forward(self, words, chars, tags=None, mask=None):
        x = self.word_emb(words)
        c = self.char_enc(chars)

        x = self.lstm(torch.cat([x, c], dim=-1))[0]
        emissions = self.fc(x)

        if tags is not None:
            return -self.crf(emissions, tags, mask)
        return self.crf.viterbi_decode(emissions, mask)
