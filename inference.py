import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from customdata import StructureDataset, get_transforms
from model_train import SeriousStabilityModel
from tqdm import tqdm

def run_ensemble():
    device = torch.device('cuda')
    _, val_trans = get_transforms(384)
    
    test_df = pd.read_csv('/home/sehoon/structure/open/sample_submission.csv')
    test_ds = StructureDataset(test_df, '/home/sehoon/structure/open/test', val_trans)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    # 5개 모델 로드
    models = []
    for i in range(5):
        m = SeriousStabilityModel().to(device)
        m.load_state_dict(torch.load(f'serious_fold_{i}.pth'))
        m.eval()
        models.append(m)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Serious Ensemble"):
            f, t = batch['front'].to(device), batch['top'].to(device)
            
            fold_probs = []
            for m in models:
                probs = torch.softmax(m(f, t), dim=1)
                fold_probs.append(probs.cpu().numpy())
            
            results.extend(np.mean(fold_probs, axis=0))

    results = np.array(results).clip(0.005, 0.995)
    test_df['unstable_prob'] = results[:, 1]
    test_df['stable_prob'] = results[:, 0]
    test_df.to_csv('saitama_submission.csv', index=False)
    print("👊 1등 뺨 때릴 준비 완료! 'saitama_submission.csv'를 확인하세요.")

if __name__ == "__main__":
    run_ensemble()