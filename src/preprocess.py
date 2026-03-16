import pandas as pd

from sklearn.preprocessing import LabelEncoder

HIGH_NULL_COLS = [
    "난자 해동 경과일", "PGS 시술 여부", "PGD 시술 여부",
    "착상 전 유전 검사 사용 여부", "임신 시도 또는 마지막 임신 경과 연수",
    "불임 원인 - 여성 요인", "불임 원인 - 정자 면역학적 요인",
]
AGE_MAP = {
    "만18-34세": 1, "만35-37세": 2, "만38-39세": 3,
    "만40-42세": 4, "만43-44세": 5, "만45-50세": 6, "알 수 없음": -1
}
DONOR_MAP = {
    "만20세 이하": 1, "만21-25세": 2, "만26-30세": 3,
    "만31-35세": 4, "만36-40세": 5, "만41-45세": 6, "알 수 없음": -1
}
CNT_MAP = {
    "0회": 0, "1회": 1, "2회": 2, "3회": 3,
    "4회": 4, "5회": 5, "6회 이상": 6
}
CNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
MALE_COLS = [
    "불임 원인 - 남성 요인", "불임 원인 - 정자 농도",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
]
FEMALE_COLS = [
    "불임 원인 - 난관 질환", "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
]
EPS = 1e-6



def load_data(train_path: str, test_path: str = None):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path) if test_path else None
    print(f"학습 데이터: {train.shape}")
    if test is not None:
        print(f"테스트 데이터: {test.shape}")
    return train, test



def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    TARGET = "임신 성공 여부"
    ID_COL = "ID"

    y = train[TARGET].copy()
    train_df = train.drop(columns=[TARGET, ID_COL])
    test_df  = test.drop(columns=[ID_COL])

    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train = len(train_df)

    
    for col in HIGH_NULL_COLS:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)
    df.drop(columns=[c for c in HIGH_NULL_COLS if c in df.columns], inplace=True)

    for col in ["배아 이식 경과일", "난자 혼합 경과일", "난자 채취 경과일", "배아 해동 경과일"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isnull().astype(int)

    df["시술 당시 나이_num"]  = df["시술 당시 나이"].map(AGE_MAP)
    df["난자 기증자 나이_num"] = df["난자 기증자 나이"].map(DONOR_MAP)
    df["정자 기증자 나이_num"] = df["정자 기증자 나이"].map(DONOR_MAP)

    for col in CNT_COLS:
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(CNT_MAP)

    # ── 파생 피처
    df["배아_이식률"]        = df["이식된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["배아_저장률"]        = df["저장된 배아 수"]              / (df["총 생성 배아 수"] + EPS)
    df["수정_성공률"]        = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + EPS)
    df["난자_활용률"]        = df["혼합된 난자 수"]             / (df["수집된 신선 난자 수"] + EPS)
    df["ICSI_이식비율"]      = df["미세주입 배아 이식 수"]       / (df["이식된 배아 수"] + EPS)
    df["전체_효율"]          = df["이식된 배아 수"]              / (df["수집된 신선 난자 수"] + EPS)
    df["배아_손실률"]        = 1 - (df["이식된 배아 수"] + df["저장된 배아 수"]) / (df["총 생성 배아 수"] + EPS)
    df["미세주입_배아_비율"] = df["미세주입에서 생성된 배아 수"] / (df["총 생성 배아 수"] + EPS)

    df["배양_기간"]           = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    df["이식_빠름"]           = (df["배아 이식 경과일"] <= 3).astype(int)
    df["이식_Day5"]           = (df["배아 이식 경과일"] == 5).astype(int)

    df["나이_x_배아수"]       = df["시술 당시 나이_num"] * df["이식된 배아 수"]
    df["나이_x_생성배아"]     = df["시술 당시 나이_num"] * df["총 생성 배아 수"]
    df["나이_x_이식경과일"]   = df["시술 당시 나이_num"] * df["배아 이식 경과일"]
    df["나이_x_시술횟수"]     = df["시술 당시 나이_num"] * df["총 시술 횟수_num"]

    df["과거_임신율"]         = df["총 임신 횟수_num"]  / (df["총 시술 횟수_num"] + EPS)
    df["임신_출산율"]         = df["총 출산 횟수_num"]  / (df["총 임신 횟수_num"] + EPS)
    df["과거_성공_경험"]      = (df["총 임신 횟수_num"] > 0).astype(int)
    df["클리닉_집중도"]       = df["클리닉 내 총 시술 횟수_num"] / (df["총 시술 횟수_num"] + EPS)

    m_cols = [c for c in MALE_COLS   if c in df.columns]
    f_cols = [c for c in FEMALE_COLS if c in df.columns]
    df["남성_불임_원인_수"]   = df[m_cols].sum(axis=1)
    df["여성_불임_원인_수"]   = df[f_cols].sum(axis=1)
    df["총_불임_원인_수"]     = df["남성_불임_원인_수"] + df["여성_불임_원인_수"]
    df["복합_불임_여부"]      = ((df["남성_불임_원인_수"] > 0) & (df["여성_불임_원인_수"] > 0)).astype(int)

    df["ICSI_포함"]           = df["특정 시술 유형"].str.contains("ICSI",       na=False).astype(int)
    df["BLASTOCYST_포함"]     = df["특정 시술 유형"].str.contains("BLASTOCYST", na=False).astype(int)
    df["AH_포함"]             = df["특정 시술 유형"].str.contains("AH",         na=False).astype(int)
    df["FER_포함"]            = df["특정 시술 유형"].str.contains("FER",        na=False).astype(int)
    df["복합시술_여부"]       = df["특정 시술 유형"].str.contains("/",          na=False).astype(int)

    df["배아_이식률_구간"]    = pd.cut(
        df["배아_이식률"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 99],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    drop_str = ["시술 당시 나이", "난자 기증자 나이", "정자 기증자 나이"] + CNT_COLS
    df.drop(columns=[c for c in drop_str if c in df.columns], inplace=True)

    # ── Label Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].fillna("missing").astype(str))

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    X      = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)

    print(f"전처리 완료 | X: {X.shape} | X_test: {X_test.shape}")
    print(f"   scale_pos_weight: {(y==0).sum()/(y==1).sum():.4f}")
    return X, X_test, y