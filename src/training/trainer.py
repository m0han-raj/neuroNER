import torch

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        words, chars, tags = zip(*batch)

        max_len = max(len(w) for w in words)

        words = torch.tensor(
            [w + [0] * (max_len - len(w)) for w in words],
            device=device
        )

        tags = torch.tensor(
            [t + [0] * (max_len - len(t)) for t in tags],
            device=device
        )

        chars = torch.tensor(
            [c + [[0]*20] * (max_len - len(c)) for c in chars],
            device=device
        )

        mask = words != 0

        optimizer.zero_grad()
        loss = model(words, chars, tags, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
