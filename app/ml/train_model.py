import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .constants import DATA_DIR, MODEL_SAVE_PATH

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_model(sample_size: int = None, epochs: int = 5):
    """
    Trains the BERT model on True.csv and Fake.csv.
    
    Args:
        sample_size (int, optional): Number of samples to use from each dataset. 
                                     Useful for quick testing. If None, uses all data.
        epochs (int): Number of training epochs. Default is 20 (Max Level).
    """
    logger.info("Starting training process...")

    # 1. Load Data
    true_csv_path = os.path.join(DATA_DIR, "True.csv")
    fake_csv_path = os.path.join(DATA_DIR, "Fake.csv")

    if not os.path.exists(true_csv_path) or not os.path.exists(fake_csv_path):
        raise FileNotFoundError(f"Data files not found in {DATA_DIR}")

    df_true = pd.read_csv(true_csv_path)
    df_fake = pd.read_csv(fake_csv_path)

    # 2. Add Labels (0 for True, 1 for Fake)
    df_true["label"] = 0
    df_fake["label"] = 1

    # 3. Combine and Shuffle
    df = pd.concat([df_true, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Use only a subset if specified (for faster testing/debugging)
    if sample_size:
        df = df.head(sample_size * 2) # * 2 because we have 2 classes roughly
        logger.info(f"Using subset of {len(df)} samples for training.")

    # 4. Prepare Data
    # prioritizing title + text for better context, or just text? 
    # Let's use 'text' column, but maybe fillna with title if empty?
    # For now, let's stick to 'text' as it usually has the article body.
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 5. Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 6. Split Data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer)

    batch_size = 16 # Reduce if OOM
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 7. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 8. Training Loop
    model.train()
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Average Training Loss: {avg_train_loss}")

    # 9. Validation (Optional but recommended to see if it works)
    model.eval()
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_true, val_preds)
    logger.info(f"Validation Accuracy: {val_acc}")

    # 10. Save Weights
    # We save the state dict so our custom classifier can load it, 
    # OR we can just save the whole model.
    # To keep it compatible with our existing BertClassifier class which uses 'bert-base-uncased',
    # we should save the state dict. 
    # However, our existing class uses `BertModel` + `nn.Linear`. 
    # `BertForSequenceClassification` wraps both. 
    # Let's save the whole `state_dict` and we might need to adjust loading logic or 
    # adjust this training script to match the architecture exactly.
    
    # Actually, simpler approach:
    # Our `FakeNewsBERT` in `bert_classifier.py` has `self.bert` and `self.classifier`.
    # `BertForSequenceClassification` also has `bert` and `classifier` (usually).
    # Let's check keys compatibility later or just save the model and load it using `BertForSequenceClassification` in the inference time too.
    # Refactoring `bert_classifier.py` to use `BertForSequenceClassification` is cleaner than manually maintaining layers.
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    
    return {"status": "success", "accuracy": val_acc, "model_path": MODEL_SAVE_PATH}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Fake News Model")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    
    args = parser.parse_args()
    
    train_model(sample_size=args.sample_size, epochs=args.epochs)
