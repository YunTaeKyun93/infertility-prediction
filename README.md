# 난임 환자 임신 성공 예측 모델

난임 환자 시술 데이터를 분석하여 임신 성공에 영향을 미치는 주요 요인을 도출하고, 최적의 AI 모델을 개발합니다.

---

## 팀 구성 및 역할

| 이름 | 역할                      | 담당 파일                         |
| ---- | ------------------------- | --------------------------------- |
| 태균 | 학습 파이프라인 설계      | `src/train.py`, `src/predict.py`  |
| 승희 | 모델 튜닝                 | Optuna 하이퍼파라미터 튜닝        |
| 재현 | EDA + Feature Engineering | `src/preprocess.py`, `notebooks/` |

---

## 폴더 구조

(예시 파일명입니다. 원하시는 대로 바꾸셔도되고 train은 2명이라 학습법을 뒤에 작성해주세요.)

```
infertility-prediction/
├── data/                   ← 각자 직접 넣기 (git 제외)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
src/
├── preprocess.py       ← 재현
├── train_lgbm.py       ← 태균
├── train_xgb.py        ← 승희
├── ensemble.py         ← 태균 (둘 합치기)
└── predict.py          ← 태균 (최종 submission)
├── notebooks/              ← EDA (재현 담당)
├── outputs/                ← 모델, submission 저장 (git 제외)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 환경 세팅

```bash
# 1. 레포 클론
git clone https://github.com/YunTaeKyun93/infertility-prediction
cd infertility-prediction

# 2. 패키지 설치
pip install -r requirements.txt

# 3. data/ 폴더에 CSV 파일 직접 넣기
# train.csv, test.csv, sample_submission.csv
```

---

## 브랜치 전략

```
main                ← 최종 제출본 (건드리지 마세요!!!!!)
└── develop         ← 팀 공유 브랜치
     ├── feat/pipeline-태균
     ├── feat/tuning-승희
     └── feat/eda-재현
```

---

## 처음 브런치 만들떄 (맨처음만 하시면됩니다!!)

# 재현

git switch develop
git switch -c feat/eda-재현
git push origin feat/eda-재현

# 승희

git switch develop
git switch -c feat/tuning-승희
git push origin feat/tuning-승희

# 태균

git switch develop
git switch -c feat/pipeline-태균
git push origin feat/pipeline-태균

````

## Git 작업 루틴

```bash
# 최우선 작업 본인 브런치로 변경하기
git switch feat/eda-재현        # 재현
git switch feat/tuning-승희     # 승희
git switch feat/pipeline-태균   # 태균

# 1. 작업 시작 전 (필수! 업데이트 안하면 무조건 오류납니다!!)
git pull origin develop

# 2. 작업 후 저장
git add .
git commit -m "feat: 작업내용"
git push origin 내브랜치이름

# 3. 작업 완료 시 GitHub에서 develop으로 PR 생성 후 태균한테 연락
````

---

## 규칙

1. main 직접 푸시 금지
2. 작업 시작 전 git pull 필수
3. 막히면 바로 태균한테 연락(아 물론 git;;; 다른건 같이 상담)

---

## 모델 전략

(해당 내용은 본인이 조사 및 생각을 하고 어떻게 왜 라는 질문에 답할수잇도록!)

```
베이스라인:
최종 모델:
평가 지표:
```

---

## 피처 엔지니어링 전략

(해당 내용은 본인이 조사 및 생각을 하고 어떻게 왜 라는 질문에 답할수잇도록!)

```
### 1. 결측치 처리

### 2. 인코딩

### 3. 파생변수

### 기타등등.. 이후 작업 내용 들어보고 수정할게요! 스케일링이나 뭐,, 그런것들
```
