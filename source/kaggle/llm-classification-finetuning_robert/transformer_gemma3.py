# ==============================================================================
# LLM Classification Fine-tuning Baseline
# Competition: https://www.kaggle.com/competitions/llm-classification-finetuning
#
# INSTRUCTIONS FOR KAGGLE:
# 1. Upload gemma-3 dataset (from HF or local) as a Kaggle dataset
# 2. Add that dataset to this notebook
# 3. Update MODEL_NAME below to match your dataset path
# 4. Enable GPU (Settings → GPU T4 x2)
# 5. Install additional libraries: !pip install peft bitsandbytes
# 6. Run the notebook
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import warnings
import os
import json
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    # UPDATE THIS: Path to your uploaded Gemma-3 dataset
    MODEL_NAME = "/kaggle/input/gemma-3/transformers/gemma-3-270m-it/1"  # 또는 "google/gemma-3-27b-it" for HF auto-download
    MAX_LENGTH = 256
    BATCH_SIZE = 16   # Reduced for memory (Gemma-3 is large)
    EPOCHS = 1        # Increased for better performance
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_CLIP = 1.0   # Gradient clipping

    # Data paths
    TRAIN_DATA_PATH = "/kaggle/input/llm-classification-finetuning/train.csv"
    TEST_DATA_PATH = "/kaggle/input/llm-classification-finetuning/test.csv"
    SAMPLE_SUBMISSION_PATH = "/kaggle/input/llm-classification-finetuning/sample_submission.csv"

    # Output paths
    SUBMISSION_PATH = "/kaggle/working/submission.csv"
    MODEL_SAVE_PATH = "/kaggle/working/best_model.pt"

config = Config()
print(f"\nConfiguration:")
print(f"  Model: {config.MODEL_NAME}")
print(f"  Device: {config.DEVICE}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Epochs: {config.EPOCHS}")

# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("Random seeds set to 42")

# ==============================================================================
# Verify Model Path
# ==============================================================================
required_files = ['config.json', 'tokenizer.json', 'pytorch_model.bin']
missing = [f for f in required_files if not any(f in x for x in os.listdir(config.MODEL_NAME))]
if missing:
    print(f"ERROR: Missing files in model dir: {missing}")
    print("Available:")
    for x in os.listdir(config.MODEL_NAME)[:10]: print("  ", x)
else:
    print("Model files OK")

# ==============================================================================
# Load Data
# ==============================================================================
train_df = pd.read_csv(config.TRAIN_DATA_PATH)
test_df = pd.read_csv(config.TEST_DATA_PATH)
sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

train_data, val_data = train_test_split(
    train_df, test_size=0.1, random_state=42,
    stratify=train_df['winner_model_a'].astype(str) + train_df['winner_model_b'].astype(str)
)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# ==============================================================================
# Dataset with Gemma-3 Chat Template (Manual)
# ==============================================================================
class LLMComparisonDataset (Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Manual chat template (Gemma-3 style)
        prompt = row['prompt']
        resp_a = row['response_a']
        resp_b = row['response_b']

        text = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n"
        text += f"<start_of_turn>model\n{resp_a}<end_of_turn>\n"
        text += f"<start_of_turn>model\n{resp_b}<end_of_turn>"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if not self.is_test:
            labels = [row['winner_model_a'], row['winner_model_b'], row['winner_tie']]
            item['labels'] = torch.tensor(np.argmax(labels), dtype=torch.long)

        return item

print("Dataset defined")

# ==============================================================================
# Custom Classifier (No bitsandbytes → Full Precision or 8bit)
# ==============================================================================
class Gemma3Classifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # float16 → bfloat16 (수치 안정성 ↑)
            device_map="auto"
        )
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        # classification head도 GPU로 이동
        self.classifier = self.classifier.to(config.DEVICE)

    def forward(self, input_ids, attention_mask):
        # 입력도 GPU로
        input_ids = input_ids.to(config.DEVICE)
        attention_mask = attention_mask.to(config.DEVICE)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        logits = self.classifier(pooled)
        
        return self.classifier(pooled)

print("Model class defined")

# ==============================================================================
# Initialize
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Gemma3Classifier(config.MODEL_NAME)
print(f"Model loaded | Params: {sum(p.numel() for p in model.parameters()):,}")

# ==============================================================================
# DataLoaders
# ==============================================================================
train_dataset = LLMComparisonDataset(train_data, tokenizer, config.MAX_LENGTH)
val_dataset = LLMComparisonDataset(val_data, tokenizer, config.MAX_LENGTH)
test_dataset = LLMComparisonDataset(test_df, tokenizer, config.MAX_LENGTH, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# ==============================================================================
# Training Functions
# ==============================================================================
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['labels'].to(config.DEVICE)

        logits = model(input_ids, attention_mask)

        # NaN 체크
        if torch.isnan(logits).any():
            print("NaN detected in logits! Skipping batch...")
            continue
        
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)

            # NaN → 0 처리
            probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
            probs = torch.clamp(probs, 1e-7, 1 - 1e-7)  # log_loss 안정화
            
            preds.append(probs.cpu().numpy())
            trues.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    trues_onehot = np.eye(3)[np.hstack(trues)]

    # NaN 제거
    preds = np.nan_to_num(preds, nan=0.0)
    
    return log_loss(trues_onehot, preds), preds

# ==============================================================================
# Optimizer
# ==============================================================================
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
total_steps = len(train_loader) * config.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

# ==============================================================================
# Training Loop
# ==============================================================================
best_val_loss = float('inf')
for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, _ = validate(model, val_loader)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print("Best model saved!")

# ==============================================================================
# Inference
# ==============================================================================
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predict"):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        predictions.append(probs.cpu().numpy())

predictions = np.vstack(predictions)
predictions = np.nan_to_num(predictions, nan=0.0)

# ==============================================================================
# Submission
# ==============================================================================

submission = sample_submission.copy()
submission['winner_model_a'] = predictions[:, 0]
submission['winner_model_b'] = predictions[:, 1]
submission['winner_tie'] = predictions[:, 2]

# 정규화 (확률 합 1)
row_sums = submission.iloc[:, 1:].sum(axis=1)
submission['winner_model_a'] /= row_sums
submission['winner_model_b'] /= row_sums
submission['winner_tie'] /= row_sums

submission.to_csv(config.SUBMISSION_PATH, index=False)

print("SUBMISSION READY!")
print(submission.head())
print("Prob sum mean:", submission.iloc[:,1:].sum(axis=1).mean())