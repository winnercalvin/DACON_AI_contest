import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
# 기본 설정
BASE_DIR = Path('opens')
IMG_SIZE = 384
BATCH_SIZE = 8
ACCUM_STEPS = 4  # effective batch = 32
BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
EPOCHS = 30
N_FOLDS = 5  # Stratified 5-Fold CV
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
train_df = pd.read_csv(BASE_DIR / 'train.csv', encoding='utf-8-sig')
dev_df = pd.read_csv(BASE_DIR / 'dev.csv', encoding='utf-8-sig')

# 주의: train은 TRAIN_0001 (4자리), dev는 DEV_001 (3자리)
train_df['front_path'] = train_df['id'].apply(lambda x: str(BASE_DIR / 'train' / x / 'front.png'))
train_df['top_path'] = train_df['id'].apply(lambda x: str(BASE_DIR / 'train' / x / 'top.png'))
dev_df['front_path'] = dev_df['id'].apply(lambda x: str(BASE_DIR / 'dev' / x / 'front.png'))
dev_df['top_path'] = dev_df['id'].apply(lambda x: str(BASE_DIR / 'dev' / x / 'top.png'))

train_df['target'] = (train_df['label'] == 'unstable').astype(int)
dev_df['target'] = (dev_df['label'] == 'unstable').astype(int)

df = pd.concat([train_df, dev_df], ignore_index=True)
print(f'Total: {len(df)}')
print(df['target'].value_counts())
test_df = pd.read_csv(BASE_DIR / 'sample_submission.csv', encoding='utf-8-sig')
test_df['front_path'] = test_df['id'].apply(lambda x: str(BASE_DIR / 'test' / x / 'front.png'))
test_df['top_path'] = test_df['id'].apply(lambda x: str(BASE_DIR / 'test' / x / 'top.png'))
print(f'Test: {len(test_df)}')

class DualStreamModel(nn.Module):
    def __init__(self, backbone_name='convnext_small.fb_in22k_ft_in1k_384', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, front, top):
        f = self.backbone(front)
        t = self.backbone(top)
        return self.head(torch.cat([f, t], dim=1)).squeeze(-1)
    
    train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                       border_mode=0, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class StructureDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        front = np.array(Image.open(row['front_path']).convert('RGB'))
        top = np.array(Image.open(row['top_path']).convert('RGB'))

        if self.transform:
            front = self.transform(image=front)['image']
            top = self.transform(image=top)['image']

        if self.is_test:
            return front, top

        target = torch.tensor(row['target'], dtype=torch.float32)
        return front, top, target
    
    def train_one_fold(fold, train_df, val_df):
    print(f'\n--- Fold {fold} ---')

    train_ds = StructureDataset(train_df, train_transform)
    val_ds = StructureDataset(val_df, val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = DualStreamModel(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': BACKBONE_LR},
        {'params': model.head.parameters(), 'lr': HEAD_LR},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    scaler = GradScaler()
    best_auc = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, (front, top, target) in enumerate(train_loader):
            front = front.to(DEVICE)
            top = top.to(DEVICE)
            target = target.to(DEVICE)

            with autocast(device_type='cuda', dtype=torch.float16):
                out = model(front, top)
                loss = criterion(out, target) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS

        scheduler.step()
        train_loss /= len(train_loader)

        # validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for front, top, target in val_loader:
                front = front.to(DEVICE)
                top = top.to(DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model(front, top)
                preds.append(torch.sigmoid(out.float()).cpu().numpy())
                targets.append(target.numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        auc = roc_auc_score(targets, preds)
        ll = log_loss(targets, np.clip(preds, 1e-7, 1 - 1e-7))

        if (epoch + 1) % 10 == 0:
            print(f'  Ep {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | AUC: {auc:.4f} | LL: {ll:.4f}')

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f'fold{fold}_best.pt')

    print(f'  Best AUC: {best_auc:.4f}')
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    train_one_fold(fold, train_data, val_data)

    def predict(test_df, fold):
    model = DualStreamModel(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(f'fold{fold}_best.pt', map_location=DEVICE, weights_only=True))
    model.eval()

    ds = StructureDataset(test_df, transform=val_transform, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)

    preds = []
    with torch.no_grad():
        for front, top in loader:
            front = front.to(DEVICE)
            top = top.to(DEVICE)
            with autocast(device_type='cuda', dtype=torch.float16):
                out = model(front, top)
            preds.append(torch.sigmoid(out.float()).cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds)

# 5-fold 평균
all_preds = []
for fold in range(N_FOLDS):
    preds = predict(test_df, fold)
    all_preds.append(preds)
    print(f'Fold {fold}: mean={preds.mean():.4f}')

final_preds = np.mean(all_preds, axis=0)
print(f'Final: mean={final_preds.mean():.4f}')

submission = test_df[['id']].copy()
submission['unstable_prob'] = np.clip(final_preds, 1e-7, 1 - 1e-7)
submission['stable_prob'] = 1.0 - submission['unstable_prob']

submission.to_csv('submission.csv', index=False, encoding='utf-8-sig')
print('submission.csv 저장 완료')
submission.head(10)