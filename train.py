import torch
from torch.utils.data import DataLoader

from src.data.dataset import TamilNERDataset
from src.data.vocab import build_vocabs
from src.models.ner_model import NERModel
from src.training.trainer import train_one_epoch
from src.training.evaluate import evaluate


# =========================
# Config
# =========================
TRAIN_PATH = "data/ta_train.json"
VAL_PATH   = "data/ta_val.json"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
MAX_LEN = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Collate Function
# =========================
def collate_fn(batch):
    words, chars, tags = zip(*batch)

    max_len = max(len(w) for w in words)
    max_char_len = 20  # Fixed character length per word

    word_tensor = torch.tensor(
        [w + [0] * (max_len - len(w)) for w in words],
        dtype=torch.long
    )

    tag_tensor = torch.tensor(
        [t + [0] * (max_len - len(t)) for t in tags],
        dtype=torch.long
    )

    # Pad each word's character sequence to max_char_len, then pad sequence to max_len
    char_list = []
    for char_seq in chars:
        padded_chars = [c + [0] * (max_char_len - len(c)) for c in char_seq]
        padded_chars += [[0] * max_char_len] * (max_len - len(char_seq))
        char_list.append(padded_chars)
    
    char_tensor = torch.tensor(char_list, dtype=torch.long)

    mask = word_tensor != 0

    return word_tensor, char_tensor, tag_tensor, mask


# =========================
# Main
# =========================
def main():
    print("🔹 Building vocabularies...")
    word_vocab, char_vocab, label_vocab = build_vocabs(TRAIN_PATH, min_freq=3, max_samples=50000)

    print(f"Words: {len(word_vocab)} | Chars: {len(char_vocab)} | Labels: {len(label_vocab)}")

    print("🔹 Loading datasets...")
    train_ds = TamilNERDataset(
        TRAIN_PATH, word_vocab, char_vocab, label_vocab, MAX_LEN, max_samples=50000
    )
    val_ds = TamilNERDataset(
        VAL_PATH, word_vocab, char_vocab, label_vocab, MAX_LEN, max_samples=10000
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    print("🔹 Initializing model...")
    model = NERModel(word_vocab, char_vocab, label_vocab).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🔹 Training on {DEVICE}\n")

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE
        )

        precision, recall, f1 = evaluate(
            model, val_loader, label_vocab, DEVICE
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"P: {precision:.4f} | "
            f"R: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "word_vocab": word_vocab,
                    "char_vocab": char_vocab,
                    "label_vocab": label_vocab,
                },
                "trained_models/best_model.pt"
            )
            print("✅ Saved best model")

    print("\n🎉 Training complete!")
    print(f"🏆 Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
