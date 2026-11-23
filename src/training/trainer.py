import torch


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        words, chars, tags, mask = batch

        words = words.to(device)
        chars = chars.to(device)
        tags = tags.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        loss = model(words, chars, tags, mask)
        loss = loss.mean()  # Average loss across batch
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
