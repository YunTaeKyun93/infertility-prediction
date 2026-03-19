"""
MLP v4 — PyTorch Neural Network
트리 모델과 완전히 다른 구조 → 앙상블 효과 극대화
BatchNorm + Dropout + 잔차 연결
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

SEED    = 42
N_FOLDS = 10
DEVICE  = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

AGE_MAP    = {"만18-34세":1,"만35-37세":2,"만38-39세":3,"만40-42세":4,"만43-44세":5,"만45-50세":6,"알 수 없음":-1}
DONOR_MAP  = {"만20세 이하":1,"만21-25세":2,"만26-30세":3,"만31-35세":4,"만36-40세":5,"만41-45세":6,"알 수 없음":-1}
CNT_MAP    = {"0회":0,"1회":1,"2회":2,"3회":3,"4회":4,"5회":5,"6회 이상":6}
CNT_COLS   = ["총 시술 횟수","클리닉 내 총 시술 횟수","IVF 시술 횟수","DI 시술 횟수",
              "총 임신 횟수","IVF 임신 횟수","DI 임신 횟수","총 출산 횟수","IVF 출산 횟수","DI 출산 횟수"]
MALE_COLS  = ["불임 원인  - 남성 요인","불임 원인 - 정자 농도","불임 원인 - 정자 운동성","불임 원인 - 정자 형태"]
FEMALE_COLS= ["불임 원인 - 난관 질환","불임 원인 - 배란 장애","불임 원인 - 자궁경부 문제","불임 원인 - 자궁내막증"]
HIGH_NULL_COLS = ["난자 해동 경과일","PGS 시술 여부","PGD 시술 여부",
                  "착상 전 유전 검사 사용 여부","임신 시도 또는 마지막 임신 경과 연수",
                  "불임 원인 - 여성 요인","불임 원인 - 정자 면역학적 요인"]
TARGET_ENCODE_COLS = ["배란 유도 유형","특정 시술 유형","난자 출처",
                      "정자 출처","시술 시기 코드","배아 생성 주요 이유"]
EPS = 1e-6


# ══════════════════════════════════
# MLP 모델 정의 (잔차 연결 + BN + Dropout)
# ══════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(x + self.block(x)))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.res1    = ResidualBlock(hidden_dims[-1], dropout)
        self.res2    = ResidualBlock(hidden_dims[-1], dropout)
        self.head    = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x).squeeze(1)


def preprocess(train, test):
    TARGET, ID_COL = "임신 성공 여부", "ID"
    y        = train[TARGET].copy()
    train_df = train.drop(columns=[TARGET, ID_COL])
    test_df  = test.drop(columns=[ID_COL])
    df       = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train  = len(train_df)

    for col in HIGH_NULL_COLS:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)
    df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns], inplace=True)

    for col in ["배아 이식 경과일","난자 혼합 경과일","난자 채취 경과일","배아 해동 경과일"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)

    df["나이"]         = df["시술 당시 나이"].map(AGE_MAP)
    df["기증난자나이"]  = df["난자 기증자 나이"].map(DONOR_MAP)
    df["기증정자나이"]  = df["정자 기증자 나이"].map(DONOR_MAP)
    for col in CNT_COLS:
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(CNT_MAP)

    # 핵심 파생 피처
    df["배아_이식률"]     = df["이식된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["배아_저장률"]     = df["저장된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["수정_성공률"]     = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + EPS)
    df["난자_활용률"]     = df["혼합된 난자 수"]             / (df["수집된 신선 난자 수"] + EPS)
    df["전체_효율"]       = df["이식된 배아 수"]              / (df["수집된 신선 난자 수"] + EPS)
    df["배아_손실률"]     = 1 - (df["이식된 배아 수"] + df["저장된 배아 수"]) / (df["총 생성 배아 수"] + EPS)
    df["배양_기간"]       = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    df["이식_Day5"]       = (df["배아 이식 경과일"] == 5).astype(int)
    df["이식_너무빠름"]   = (df["배아 이식 경과일"] <= 2).astype(int)
    df["나이_x_배아수"]   = df["나이"] * df["이식된 배아 수"]
    df["나이_x_생성배아"] = df["나이"] * df["총 생성 배아 수"]
    df["나이_x_Day5"]     = df["나이"] * df["이식_Day5"]
    df["나이_제곱"]       = df["나이"] ** 2
    df["과거_임신율"]     = df["총 임신 횟수_n"]  / (df["총 시술 횟수_n"] + EPS)
    df["임신_출산율"]     = df["총 출산 횟수_n"]  / (df["총 임신 횟수_n"] + EPS)
    df["과거_성공_경험"]  = (df["총 임신 횟수_n"] > 0).astype(int)
    df["유산_비율"]       = (df["총 임신 횟수_n"] - df["총 출산 횟수_n"]) / (df["총 임신 횟수_n"] + EPS)
    df["출산_성공율"]     = df["총 출산 횟수_n"]  / (df["총 시술 횟수_n"] + EPS)
    df["잔여_배아_수"]    = df["총 생성 배아 수"] - df["이식된 배아 수"]
    df["IVF_비율"]        = (df["IVF 시술 횟수_n"] / (df["총 시술 횟수_n"] + EPS)).clip(0,1)
    df["IVF_임신율"]      = (df["IVF 임신 횟수_n"] / (df["IVF 시술 횟수_n"] + EPS)).clip(0,10)
    df["DI_임신율"]       = (df["DI 임신 횟수_n"]  / (df["DI 시술 횟수_n"] + EPS)).clip(0,10)
    df["클리닉_경험_비율"] = df["클리닉 내 총 시술 횟수_n"] / (df["총 시술 횟수_n"] + EPS)

    m_cols = [c for c in MALE_COLS   if c in df.columns]
    f_cols = [c for c in FEMALE_COLS if c in df.columns]
    df["남성_원인_수"]    = df[m_cols].sum(axis=1)
    df["여성_원인_수"]    = df[f_cols].sum(axis=1)
    df["총_원인_수"]      = df["남성_원인_수"] + df["여성_원인_수"]
    df["복합_불임"]       = ((df["남성_원인_수"]>0)&(df["여성_원인_수"]>0)).astype(int)
    df["원인_없음"]       = (df["총_원인_수"]==0).astype(int)
    df["나이_x_불임원인"] = df["나이"] * df["총_원인_수"]

    df["ICSI_포함"]       = df["특정 시술 유형"].str.contains("ICSI",       na=False).astype(int)
    df["BLAST_포함"]      = df["특정 시술 유형"].str.contains("BLASTOCYST", na=False).astype(int)
    df["기증_난자"]       = (df["기증난자나이"]>0).astype(int)
    df["기증_정자"]       = (df["기증정자나이"]>0).astype(int)
    df["고령_저배아"]     = ((df["나이"]>=4)&(df["이식된 배아 수"]<=1)).astype(int)
    df["반복실패"]        = (df["총 시술 횟수_n"]>=3).astype(int)
    df["저품질_배아"]     = (df["배아_이식률"]>=0.9).astype(int)
    df["저반응_난소"]     = (df["총 생성 배아 수"]<=2).astype(int)
    df["고령_반복실패"]   = ((df["나이"]>=4)&(df["총 시술 횟수_n"]>=3)).astype(int)
    df["적정_난자수"]     = ((df["수집된 신선 난자 수"]>=8)&(df["수집된 신선 난자 수"]<=14)).astype(int)
    df["난소_과자극"]     = (df["수집된 신선 난자 수"]>=15).astype(int)

    df["배아_품질_점수"]  = (df["이식_Day5"]*4 + df["적정_난자수"]*2 +
                             (1-df["저품질_배아"])*2 + (1-df["저반응_난소"]) +
                             df["기증_난자"]*2)
    df["실패_위험_점수"]  = (df["고령_저배아"]*3 + df["반복실패"]*2 +
                             df["이식_너무빠름"]*2 + df["저반응_난소"] + df["저품질_배아"])

    sperm_c = [c for c in ["불임 원인 - 정자 농도","불임 원인 - 정자 운동성","불임 원인 - 정자 형태"] if c in df.columns]
    df["정자_문제_수"]    = df[sperm_c].sum(axis=1)

    # inf 처리
    for col in df.select_dtypes(include=["float64","int64"]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    drop_str = ["시술 당시 나이","난자 기증자 나이","정자 기증자 나이"] + CNT_COLS
    df.drop(columns=[c for c in drop_str if c in df.columns], inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].fillna("missing").astype(str))
    for col in df.select_dtypes(include=["float64","int64"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    X      = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)

    # K-Fold 타깃 인코딩
    global_mean = y.mean()
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for col in TARGET_ENCODE_COLS:
        if col not in X.columns: continue
        oof_te = np.full(len(X), global_mean)
        for tr_idx, val_idx in skf5.split(X, y):
            mm = y.iloc[tr_idx].groupby(X[col].iloc[tr_idx]).mean()
            oof_te[val_idx] = X[col].iloc[val_idx].map(mm).fillna(global_mean).values
        X[f"{col}_te"]      = oof_te
        X_test[f"{col}_te"] = X_test[col].map(y.groupby(X[col]).mean()).fillna(global_mean)

    print(f"✅ 전처리 완료 | X:{X.shape}")
    return X, X_test, y


def train_mlp_fold(X_tr, y_tr, X_val, y_val, X_test, input_dim, pos_weight):
    model = MLP(input_dim, hidden_dims=[512, 256, 128], dropout=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )

    # 데이터 준비
    X_tr_t  = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    X_tst_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=2048, shuffle=True)

    best_auc, best_val_pred, best_test_pred = 0, None, None
    patience, no_improve = 20, 0

    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred  = torch.sigmoid(model(X_val_t)).cpu().numpy()
            test_pred = torch.sigmoid(model(X_tst_t)).cpu().numpy()

        auc = roc_auc_score(y_val, val_pred)
        if auc > best_auc:
            best_auc       = auc
            best_val_pred  = val_pred
            best_test_pred = test_pred
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_pred, best_test_pred, best_auc


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df  = pd.read_csv("data/test.csv")
    print(f"✅ 학습: {train_df.shape} | 테스트: {test_df.shape}")

    X, X_test, y = preprocess(train_df, test_df)

    # StandardScaler (MLP는 정규화 필수)
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)
    Xt_scaled = scaler.transform(X_test)

    pos_weight = (y==0).sum() / (y==1).sum()
    print(f"pos_weight: {pos_weight:.4f}")

    # K-Fold 학습
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X))
    tst_pred = np.zeros(len(X_test))

    print(f"\n🧠 MLP {N_FOLDS}-Fold 학습...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_tr, X_val   = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val_f = y.iloc[tr_idx].values, y.iloc[val_idx].values

        val_p, test_p, auc = train_mlp_fold(
            X_tr, y_tr, X_val, y_val_f, Xt_scaled,
            input_dim=X_scaled.shape[1], pos_weight=pos_weight
        )
        oof_pred[val_idx] = val_p
        tst_pred          += test_p / N_FOLDS
        print(f"  Fold{fold+1:2d}: {auc:.5f}")

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"\n  ➜ MLP OOF: {oof_auc:.5f}")

    # 저장
    ids = test_df["ID"]
    pd.DataFrame({"ID":ids,"probability":tst_pred}).to_csv(
        "submission_mlp.csv", index=False)

    print(f"\n🏆 submission_mlp.csv 저장 완료")
    print(f"   OOF AUC: {oof_auc:.5f}")
    print(f"   mean={tst_pred.mean():.4f} | max={tst_pred.max():.4f}")
