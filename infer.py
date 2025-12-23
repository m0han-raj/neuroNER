import torch
import sys

from src.models.ner_model import NERModel


def predict_text(text, model, word_vocab, char_vocab, label_vocab, device, max_len=80):
    """
    Predict NER tags for input text.
    
    Args:
        text: Input text string (Tamil)
        model: Trained NER model
        word_vocab: Word vocabulary dict
        char_vocab: Character vocabulary dict
        label_vocab: Label vocabulary dict
        device: Device to run inference on
        max_len: Maximum sequence length
    
    Returns:
        List of (word, tag) tuples
    """
    model.eval()
    
    # Tokenize (simple whitespace split)
    words = text.strip().split()[:max_len]
    
    if not words:
        return []
    
    # Convert to vocab indices
    word_ids = [word_vocab.get(w, 1) for w in words]  # 1 = <UNK>
    char_ids = [[char_vocab.get(c, 1) for c in w[:20]] for w in words]
    
    # Pad character sequences
    max_char_len = 20
    char_ids_padded = [cid + [0] * (max_char_len - len(cid)) for cid in char_ids]
    
    # Create tensors
    word_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)
    char_tensor = torch.tensor([char_ids_padded], dtype=torch.long).to(device)
    mask = torch.ones_like(word_tensor, dtype=torch.bool).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(word_tensor, char_tensor, mask=mask)
    
    # Convert predictions to labels
    id2label = {v: k for k, v in label_vocab.items()}
    pred_tags = [id2label[p] for p in predictions[0]]
    
    return list(zip(words, pred_tags))


def format_entities(word_tag_pairs):
    """
    Format predictions with entity highlighting.
    """
    print("\n" + "="*60)
    print("PREDICTIONS:")
    print("="*60)
    
    entities = []
    current_entity = []
    current_type = None
    
    for word, tag in word_tag_pairs:
        if tag.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = [word]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = []
                current_type = None
    
    if current_entity:
        entities.append((current_entity, current_type))
    
    # Print word-by-word
    print("\nWord-by-word:")
    for word, tag in word_tag_pairs:
        print(f"  {word:<30} → {tag}")
    
    # Print entities
    if entities:
        print("\nExtracted Entities:")
        for entity_words, entity_type in entities:
            entity_text = " ".join(entity_words)
            print(f"  [{entity_type}] {entity_text}")
    else:
        print("\nNo entities found.")
    
    print("="*60 + "\n")


def main():
    # Configuration
    MODEL_PATH = "trained_models/best_model.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔹 Loading model from {MODEL_PATH}...")
    print(f"🔹 Using device: {DEVICE}")
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Train the model first using: python train.py")
        sys.exit(1)
    
    word_vocab = checkpoint["word_vocab"]
    char_vocab = checkpoint["char_vocab"]
    label_vocab = checkpoint["label_vocab"]
    
    print(f"📊 Vocab sizes - Words: {len(word_vocab)} | Chars: {len(char_vocab)} | Labels: {len(label_vocab)}")
    
    # Initialize model
    model = NERModel(word_vocab, char_vocab, label_vocab).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print("✅ Model loaded successfully!\n")
    
    # Interactive mode or single prediction
    if len(sys.argv) > 1:
        # Single prediction from command line
        text = " ".join(sys.argv[1:])
        predictions = predict_text(text, model, word_vocab, char_vocab, label_vocab, DEVICE)
        format_entities(predictions)
    else:
        # Interactive mode
        print("="*60)
        print("Interactive NER Prediction Mode")
        print("Type Tamil text to get entity predictions")
        print("Type 'quit' or 'exit' to stop")
        print("="*60 + "\n")
        
        while True:
            try:
                text = input("Enter text: ").strip()
                
                if text.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    continue
                
                predictions = predict_text(text, model, word_vocab, char_vocab, label_vocab, DEVICE)
                format_entities(predictions)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
