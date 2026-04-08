import pandas as pd
import matplotlib
# 화면을 띄우지 않고 파일 저장용 백엔드 사용 설정 (에러 방지 핵심!)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cv2
import os

# 1. 경로 설정
BASE_PATH = '/home/sehoon/structure/open'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TRAIN_DIR = os.path.join(BASE_PATH, 'train')

def compare_stability(stable_id, unstable_id):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def load_img(path):
        if not os.path.exists(path):
            print(f"⚠️ 파일을 찾을 수 없습니다: {path}")
            return None
        img = cv2.imread(path)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 경로 설정
    paths = [
        (os.path.join(TRAIN_DIR, str(stable_id), 'front.png'), f'Stable Front ({stable_id})'),
        (os.path.join(TRAIN_DIR, str(stable_id), 'top.png'), 'Stable Top'),
        (os.path.join(TRAIN_DIR, str(unstable_id), 'front.png'), f'Unstable Front ({unstable_id})'),
        (os.path.join(TRAIN_DIR, str(unstable_id), 'top.png'), 'Unstable Top')
    ]

    for ax, (path, title) in zip(axes.flatten(), paths):
        img = load_img(path)
        if img is not None:
            ax.imshow(img)
            ax.set_title(title)
        else:
            ax.set_title(f"Error: {title}")

    plt.tight_layout()
    
    # 🔥 중요: 화면에 띄우는 대신 파일로 저장!
    save_name = 'analysis_result.png'
    plt.savefig(save_name)
    print(f"✅ 분석 결과가 '{os.getcwd()}/{save_name}'에 저장되었습니다.")

if __name__ == "__main__":
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ 에러: {TRAIN_CSV} 파일이 없습니다. 경로를 확인하세요.")
    else:
        train_df = pd.read_csv(TRAIN_CSV)
        
        # label 컬럼 확인 (데이터가 문자열인지 숫자인지 체크)
        stable_sample = train_df[train_df['label'] == 'stable'].iloc[0]['id']
        unstable_sample = train_df[train_df['label'] == 'unstable'].iloc[0]['id']
        
        compare_stability(stable_sample, unstable_sample)