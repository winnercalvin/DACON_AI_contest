# DACON 구조물 안정성 물리 추론 AI 경진대회

**대회 링크:** https://dacon.io/competitions/official/236686  
**기간:** 2026.03.03 ~ 2026.03.30  
**팀:** 승호Calvin (3인)

구조물의 front/top 이미지 쌍을 입력받아 10초 시뮬레이션 후 **stable / unstable** 여부를 예측하는 이진 분류 태스크입니다.  
평가 지표: **Log Loss**

---

## 최종 결과

| 구분 | 점수 |
|------|------|
| Private LB (최종) | **0.05815** |
| 순위 | 82위 / 866팀 |
| 제출 횟수 | 27회 |

---

## 데이터

| 분류 | 설명 |
|------|------|
| train | 1,000개 샘플 (front.png + top.png + simulation.mp4, 고정 카메라/조명) |
| dev | 100개 샘플 (front.png + top.png + label, 랜덤 카메라/조명) |
| test | 1,000개 샘플 (front.png + top.png, 랜덤 카메라/조명) |

> **핵심 도전**: Train은 고정 카메라, Dev/Test는 랜덤 카메라 + zoom-in → 도메인 갭 존재

---

## 모델 구조

### SeriousStabilityModel (메인)
`model_train.py` / `customdata.py` / `inference.py`

ConvNeXtV2-Base 백본에 SE Block과 Triple Fusion을 결합한 구조입니다.

```
front.png → ConvNeXtV2-Base → SEBlock ─┐
                                        ├→ [sum, |diff|, product] → Linear Head → logit
top.png   → ConvNeXtV2-Base → SEBlock ─┘
```

- **Backbone:** `convnextv2_base.fcmae_ft_in22k_in1k_384` (feat_dim=1024)
- **Fusion:** Triple Fusion (합/차/곱) → Linear(3072→512) → Linear(512→2)
- **학습:** Stratified 5-Fold CV, AMP(fp16), Gradient Accumulation (effective batch=32)
- **정규화:** Label Smoothing 0.05, Dropout 0.3

### DualStreamModel (초기 실험)
`1stnom.py`

ConvNext-Small 백본에 단순 concat fusion을 사용한 초기 베이스라인입니다.

```
front.png → ConvNext-Small ─┐
                              ├→ [concat] → Linear Head → logit
top.png   → ConvNext-Small ─┘
```

- **Backbone:** `convnext_small.fb_in22k_ft_in1k_384`
- **학습:** 30 epochs, CosineAnnealingWarmRestarts, AUC 기준 best model 저장

---

## 파일 구조

```
DACON_AI_contest/
├── customdata.py       # Dataset 클래스 및 Albumentations augmentation
├── model_train.py      # SeriousStabilityModel 학습 (5-Fold CV)
├── inference.py        # 5-fold 앙상블 추론 → submission CSV 생성
├── 1stnom.py           # DualStreamModel 초기 베이스라인
└── visual.py           # EDA: stable/unstable 샘플 비교 시각화
```

---

## 실행 방법

### 데이터 준비
```
opens/
├── train.csv
├── dev.csv
├── sample_submission.csv
├── train/TRAIN_0001/{front.png, top.png}
├── dev/DEV_001/{front.png, top.png}
└── test/TEST_0001/{front.png, top.png}
```

### 학습
```bash
python model_train.py
# 각 fold별 best model → serious_fold_{0~4}.pth 저장
```

### 추론 (앙상블)
```bash
python inference.py
# 5-fold 앙상블 → saitama_submission.csv 생성
```

### EDA 시각화
```bash
python visual.py
# stable/unstable 비교 이미지 → analysis_result.png 저장
```

---

## 환경

```
torch >= 2.0
timm
albumentations
scikit-learn
pandas, numpy
opencv-python
```

```bash
pip install torch timm albumentations scikit-learn pandas numpy opencv-python
```
