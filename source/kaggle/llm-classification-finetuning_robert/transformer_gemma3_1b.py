# ==============================================================================
# LLM Classification Fine-tuning Baseline (bitsandbytes 대체: FP16 + LoRA)
# Competition: https://www.kaggle.com/competitions/llm-classification-finetuning
#
# 개선 포인트 (bitsandbytes 없이):
# - torch_dtype=torch.float16: Half-precision 로드 (메모리 50% ↓)
# - Gradient Checkpointing: 메모리 절감
# - LoRA (PEFT): 학습 파라미터 최소화
# - Mixed Precision: torch.amp.autocast
# - Gradient Accumulation: steps=4 (효과적 배치=16)
# - Optimizer: 표준 AdamW
#
# peft 설치: 오프라인 데이터셋 추가 후 !pip install /path/to/peft.whl (e.g., nbroad/peft-main)
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
import gc
warnings.filterwarnings('ignore')

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    MODEL_NAME = "/kaggle/input/gemma-3/transformers/gemma-3-1b-pt/1"
    MAX_LENGTH = 128    # 256
    BATCH_SIZE = 4      # OOM 방지
    ACCUMULATION_STEPS = 4  # 효과적 배치=16
    EPOCHS = 1
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_CLIP = 1.0

    TRAIN_DATA_PATH = "/kaggle/input/llm-classification-finetuning/train.csv"
    TEST_DATA_PATH = "/kaggle/input/llm-classification-finetuning/test.csv"
    SAMPLE_SUBMISSION_PATH = "/kaggle/input/llm-classification-finetuning/sample_submission.csv"

    SUBMISSION_PATH = "/kaggle/working/submission.csv"
    MODEL_SAVE_PATH = "/kaggle/working/best_model.pt"

config = Config()
print(f"\nConfiguration (No bitsandbytes, FP16 optimized):")
print(f"  Model: {config.MODEL_NAME}")
print(f"  Device: {config.DEVICE}")
print(f"  Batch Size: {config.BATCH_SIZE} (with accumulation {config.ACCUMULATION_STEPS})")

# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
# Dataset with Gemma-3 Chat Template
# ==============================================================================
class LLMComparisonDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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

# ==============================================================================
# Custom Classifier with LoRA + FP16
# ==============================================================================
class Gemma3Classifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # FP16 대체 (메모리 절감)
            device_map="auto"
        )
        
        # Gradient Checkpointing
        self.base_model.gradient_checkpointing_enable()
        
        # LoRA Config (bitsandbytes 없이 표준 LoRA)
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Gemma-3 적합
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes).to(config.DEVICE)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)

print("Model class defined (LoRA + FP16 + Checkpointing)")

# ==============================================================================
# Initialize
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Gemma3Classifier(config.MODEL_NAME)
print(f"Model loaded | Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
# Training Functions (Mixed Precision + Accumulation)
# ==============================================================================
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler()  # Mixed Precision

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    accum_steps = 0
    
    for batch in tqdm(loader, desc="Train"):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Mixed Precision (FP16)
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / config.ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        accum_steps += 1

        if accum_steps % config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * config.ACCUMULATION_STEPS

        # 메모리 정리
        del input_ids, attention_mask, labels, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(loader, desc="Val"):
                input_ids = batch['input_ids'].to(config.DEVICE)
                attention_mask = batch['attention_mask'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)

                logits = model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

                preds.append(probs.cpu().numpy())
                trues.append(labels.cpu().numpy())

    preds = np.vstack(preds)
    trues_onehot = np.eye(3)[np.hstack(trues)]
    return log_loss(trues_onehot, preds), preds

# ==============================================================================
# Optimizer (표준 AdamW)
# ==============================================================================
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
total_steps = (len(train_loader) // config.ACCUMULATION_STEPS) * config.EPOCHS
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
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for batch in tqdm(test_loader, desc="Predict"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
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

row_sums = submission.iloc[:, 1:].sum(axis=1) + 1e-8
submission['winner_model_a'] /= row_sums
submission['winner_model_b'] /= row_sums
submission['winner_tie'] /= row_sums

submission.to_csv(config.SUBMISSION_PATH, index=False)

print("SUBMISSION READY!")
print(submission.head())
print("Prob sum mean:", submission.iloc[:,1:].sum(axis=1).mean())