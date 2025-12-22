from TorchCRF import CRF
import torch.nn as nn

class CRFLayer(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask)

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask)
