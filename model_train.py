import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from customdata import StructureDataset, get_transforms
from tqdm import tqdm

# --- [모델 구조: Serious Series Triple Fusion] ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return x * y

class SeriousStabilityModel(nn.Module):
    def __init__(self, model_name='convnextv2_base.fcmae_ft_in22k_in1k_384'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features # Base = 1024
        
        self.front_se = SEBlock(feat_dim)
        self.top_se = SEBlock(feat_dim)

        # 트리플 퓨전 헤드 (합, 차, 곱)
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, front, top):
        f = self.backbone(front)
        t = self.backbone(top)
        f, t = self.front_se(f), self.top_se(t)

        # ⭐ 사이타마급 퓨전 로직
        sum_f = f + t
        diff_f = torch.abs(f - t)
        prod_f = f * t
        
        combined = torch.cat([sum_f, diff_f, prod_f], dim=1)
        return self.head(combined)

def train_fold(fold, train_df, val_df, device):
    IMG_SIZE = 384
    BATCH_SIZE = 2         # VRAM 8GB 최적화
    ACCUM_STEPS = 16       # Effective Batch = 32
    EPOCHS = 15
    
    train_trans, val_trans = get_transforms(IMG_SIZE)
    train_ds = StructureDataset(train_df, '/home/sehoon/structure/open/train', train_trans)
    val_ds = StructureDataset(val_df, '/home/sehoon/structure/open/train', val_trans)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SeriousStabilityModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold} Ep {epoch+1}")):
            f, t, l = batch['front'].to(device), batch['top'].to(device), batch['label'].to(device)
            with torch.amp.autocast('cuda'):
                out = model(f, t)
                loss = criterion(out, l) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i+1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                f, t, l = batch['front'].to(device), batch['top'].to(device), batch['label'].to(device)
                out = model(f, t)
                v_loss += criterion(out, l).item()
        
        avg_loss = v_loss / len(val_loader)
        print(f"✅ Fold {fold} Val Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'serious_fold_{fold}.pth')
        scheduler.step()

if __name__ == "__main__":
    df = pd.read_csv('/home/sehoon/structure/open/train.csv')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (t_idx, v_idx) in enumerate(skf.split(df, df['label'])):
        train_fold(fold, df.iloc[t_idx], df.iloc[v_idx], torch.device('cuda'))