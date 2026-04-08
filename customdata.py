import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class StructureDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'stable': 0, 'unstable': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        
        # 이미지 로드 (Front & Top)
        f_path = os.path.join(self.img_dir, sample_id, 'front.png')
        t_path = os.path.join(self.img_dir, sample_id, 'top.png')
        
        front_img = cv2.imread(f_path)
        top_img = cv2.imread(t_path)
        
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        top_img = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=front_img, image_top=top_img)
            front_img = augmented['image']
            top_img = augmented['image_top']

        # target 설정 (훈련 시에만 사용)
        if 'label' in row:
            label = self.label_map[row['label']]
            return {'front': front_img, 'top': top_img, 'label': torch.tensor(label, dtype=torch.long)}
        else:
            return {'id': sample_id, 'front': front_img, 'top': top_img}

def get_transforms(img_size=384):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Perspective(scale=(0.05, 0.08), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        # 0.1점대 파괴용 Cutout
        A.CoarseDropout(max_holes=8, max_height=40, max_width=40, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'image_top': 'image'})

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'image_top': 'image'})
    
    return train_transform, val_transform