# Neural Named Entity Recognition

A BiLSTM-CRF model for Named Entity Recognition in Tamil text using character-level and word-level embeddings.

![Demo](media/demo.gif)

## Project Structure

```
neuroNER/
├── data/
│   ├── ta_train.json       # Training data
│   ├── ta_val.json         # Validation data
│   └── ta_test.json        # Test data
├── src/
│   ├── data/
│   │   ├── dataset.py      # Dataset loader
│   │   └── vocab.py        # Vocabulary builder
│   ├── models/
│   │   ├── ner_model.py    # Main NER model
│   │   ├── char_cnn.py     # Character CNN encoder
│   │   └── crf.py          # CRF layer wrapper
│   └── training/
│       ├── trainer.py      # Training loop
│       └── evaluate.py     # Evaluation metrics
├── trained_models/
│   └── best_model.pt       # Saved model checkpoint
├── train.py                # Training script
├── infer.py                # Inference script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Entity Types

Tags used:
- `B-PER` / `I-PER`: Person names
- `B-LOC` / `I-LOC`: Locations
- `B-ORG` / `I-ORG`: Organizations
- `O`: Outside entity (not an entity)

## Model Architecture

```
Input: Tamil Text
    ↓
Word Embeddings (100d) + CharCNN (50d)
    ↓
BiLSTM (256d hidden)
    ↓
Linear Layer
    ↓
CRF Layer
    ↓
Output: NER Tags (B-PER, I-PER, B-LOC, I-LOC, etc.)
```

**Components:**
- Word embeddings: 100 dimensions
- Character CNN: Processes up to 20 chars per word → 50d output
- BiLSTM: 128 hidden units (256 bidirectional)
- CRF: For sequence-level tag prediction

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python 3.8+
- PyTorch
- TorchCRF
- numpy, tqdm

## Dataset Format

Data files should be in JSON Lines format (`.json`):

```json
{"words": ["வார்த்தை1", "வார்த்தை2", "..."], "ner": ["B-PER", "O", "..."]}
{"words": ["வார்த்தை3", "வார்த்தை4", "..."], "ner": ["B-LOC", "I-LOC", "..."]}
```

**Required Files:**
- `data/ta_train.json` - Training data
- `data/ta_val.json` - Validation data
- `data/ta_test.json` - Test data (optional)

**Dataset Source:**

This project uses the [Naamapadam dataset](https://huggingface.co/datasets/ai4bharat/naamapadam) from AI4Bharat, a comprehensive Indian Named Entity Recognition dataset.

## Setup

1. **Clone or download the project**
    ```bash
    git clone https://github.com/m0han-raj/neuroNER.git
    cd neuroNER
    ```
2. **Create virtual environment** (recommended):
   ```bash
   python -m venv env
   # Windows
   .\env\Scripts\activate
   # Linux/Mac
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create output directory**:
   ```bash
   mkdir trained_models
   ```

## Training

### Quick Start 

Train on 50K sentences for faster experimentation:

```bash
python train.py
```

**Default Configuration:**
- Training sentences: 50,000
- Validation sentences: 10,000
- Batch size: 16
- Epochs: 20
- Learning rate: 0.001
- Max sequence length: 80 tokens

**Expected Training Time:**
- T4 GPU: ~20-40 minutes
- CPU: ~2-4 hours

**Full Dataset Training**

To train on the full dataset, modify `train.py`:

```python
train_ds = TamilNERDataset(
    TRAIN_PATH, word_vocab, char_vocab, label_vocab, MAX_LEN = #MAX SIZE
)
```

## Inference

### Interactive Mode

Run predictions interactively:

```bash
python infer.py
```

Then type Tamil text and press Enter to get predictions.

### Command Line Prediction

Predict from command line:

```bash
python infer.py "சீதா தமிழ்நாடு மாநிலத்தில் வசிக்கிறார்"
```

## Configuration

Edit `train.py` to modify hyperparameters:

```python
BATCH_SIZE = 16     
EPOCHS = 20          
LR = 1e-3           
MAX_LEN = 80       
```
Model will be saved in `trained_models/best_model.pt`

## References

This project is inspired by NeuroNER:

```bibtex
@article{dernoncourt2017neuroner,
  title= {NeuroNER: an easy-to-use program for named-entity recognition based on neural networks},
  author= {Franck Dernoncourt and Ji Young Lee and Peter Szolovits},
  journal= {arXiv preprint arXiv:1705.05487},
  year= {2017},
  url= {https://arxiv.org/abs/1705.05487}
}
```

## License

MIT. See [LICENSE](LICENSE).
