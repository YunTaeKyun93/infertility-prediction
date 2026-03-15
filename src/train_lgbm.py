# 저랑 승희님

# import pandas as pd

# # 데이터 읽기 (경로를 정확히 입력하세요)
# df = pd.read_csv('train.csv')

# # 1. 상위 5개 행 보기 (데이터가 어떻게 생겼는지)
# print(df.head())

# # 2. 컬럼별 정보 확인 (데이터 타입, 결측치 여부)
# print(df.info())

# # 3. 통계적 정보 확인 (평균, 최대, 최소, 분위수 등)
# print(df.describe())


# import numpy as np
# import lightgbm as lgb
# import shap
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")

# # ── 한글 폰트 설정 ────────────────────────────────────────────────
# plt.rcParams["font.family"] = "AppleGothic"   # Mac
# # plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
# plt.rcParams["axes.unicode_minus"] = False


# # ════════════════════════════════════════════════════════
# # 1. 데이터 로드
# # ════════════════════════════════════════════════════════
# def load_data(train_path: str, test_path: str = None):
#     train = pd.read_csv(train_path)
#     test  = pd.read_csv(test_path) if test_path else None
#     print(f"✅ 학습 데이터: {train.shape}")
#     if test is not None:
#         print(f"✅ 테스트 데이터: {test.shape}")
#     return train, test


# # ════════════════════════════════════════════════════════
# # 2. 전처리
# # ════════════════════════════════════════════════════════
# # 결측률이 너무 높아 제거할 컬럼 (95% 이상)
# DROP_COLS = [
#     "착상 전 유전 검사 사용 여부",  # 99% 결측
#     "PGD 시술 여부",               # 99% 결측
#     "PGS 시술 여부",               # 99% 결측
#     "난자 해동 경과일",             # 99% 결측
# ]

# # 나이 순서 매핑
# AGE_ORDER = {
#     "만18-34세": 0,
#     "만35-37세": 1,
#     "만38-39세": 2,
#     "만40-42세": 3,
#     "만43-44세": 4,
#     "만45-50세": 5,
# }

# # 횟수 컬럼 순서 매핑 (예: "1회" → 1)
# COUNT_COLS = [
#     "총 시술 횟수", "클리닉 내 총 시술 횟수",
#     "IVF 시술 횟수", "DI 시술 횟수",
#     "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
#     "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
# ]

# def parse_count(val):
#     """'1회', '2회', '3회 이상' 등 → 숫자"""
#     if pd.isna(val):
#         return np.nan
#     val = str(val).strip()
#     if "이상" in val:
#         return int(val[0]) + 1   # 보수적으로 +1
#     digits = "".join(filter(str.isdigit, val))
#     return int(digits) if digits else np.nan


# def preprocess(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
#     df = df.copy()

#     # ── 타깃 분리 ──────────────────────────────────────
#     target = None
#     if is_train and "임신 성공 여부" in df.columns:
#         target = df.pop("임신 성공 여부")

#     # ── ID 제거 ────────────────────────────────────────
#     df.drop(columns=["ID"], errors="ignore", inplace=True)

#     # ── 고결측 컬럼 제거 ───────────────────────────────
#     df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

#     # ── 나이 Ordinal 인코딩 ────────────────────────────
#     df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

#     # ── 횟수 컬럼 수치화 ───────────────────────────────
#     for col in COUNT_COLS:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_count)

#     # ── 결측 여부 이진 피처 추가 ───────────────────────
#     # (결측 자체가 의미 있는 컬럼)
#     missing_flag_cols = ["임신 시도 또는 마지막 임신 경과 연수"]
#     for col in missing_flag_cols:
#         if col in df.columns:
#             df[f"{col}_결측"] = df[col].isna().astype(int)

#     # ── 범주형 Label Encoding ──────────────────────────
#     cat_cols = df.select_dtypes(include="object").columns.tolist()

#     if encoders is None:
#         encoders = {}

#     for col in cat_cols:
#         if is_train:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             encoders[col] = le
#         else:
#             le = encoders.get(col)
#             if le:
#                 df[col] = df[col].astype(str).apply(
#                     lambda x: le.transform([x])[0] if x in le.classes_ else -1
#                 )

#     # ── 수치형 결측값 중앙값 대체 ─────────────────────
#     num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
#     if is_train:
#         medians = df[num_cols].median()
#         encoders["medians"] = medians
#     else:
#         medians = encoders.get("medians", df[num_cols].median())

#     df[num_cols] = df[num_cols].fillna(medians)

#     print(f"✅ 전처리 완료: {df.shape} | 결측값: {df.isna().sum().sum()}")
#     return df, target, encoders


# # ════════════════════════════════════════════════════════
# # 3. LightGBM 학습
# # ════════════════════════════════════════════════════════
# def train_lgb(X_train, y_train, X_val, y_val):
#     params = {
#         "objective":        "binary",
#         "metric":           "auc",
#         "learning_rate":    0.05,
#         "num_leaves":       127,
#         "max_depth":        -1,
#         "min_child_samples": 50,
#         "subsample":        0.8,
#         "colsample_bytree": 0.8,
#         "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),  # 불균형 보정
#         "n_estimators":     1000,
#         "random_state":     42,
#         "verbose":          -1,
#     }

#     model = lgb.LGBMClassifier(**params)
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
#     )

#     val_pred = model.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, val_pred)
#     print(f"\n✅ Validation AUC: {auc:.4f}")
#     print(classification_report(y_val, (val_pred > 0.5).astype(int), target_names=["실패", "성공"]))
#     return model


# # ════════════════════════════════════════════════════════
# # 4. SHAP 분석
# # ════════════════════════════════════════════════════════
# def analyze_shap(model, X_val: pd.DataFrame, save_dir: str = "."):
#     print("\n🔍 SHAP 분석 시작...")
#     explainer   = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_val)

#     # LightGBM binary: shap_values가 list면 [1] 인덱스 (성공 클래스)
#     if isinstance(shap_values, list):
#         sv = shap_values[1]
#     else:
#         sv = shap_values

#     # ── ① Summary Plot (전체 피처 중요도) ─────────────
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, show=False, max_display=20)
#     plt.title("SHAP Summary Plot — 피처별 임신 성공 영향도", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 Summary Plot 저장: {save_dir}/shap_summary.png")

#     # ── ② Bar Plot (평균 절댓값 중요도) ───────────────
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
#     plt.title("SHAP Feature Importance (평균 |SHAP|)", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 Importance Plot 저장: {save_dir}/shap_importance.png")

#     # ── ③ 상위 피처 텍스트 출력 ───────────────────────
#     mean_shap = pd.Series(
#         np.abs(sv).mean(axis=0),
#         index=X_val.columns
#     ).sort_values(ascending=False)

#     print("\n📋 상위 10개 중요 피처:")
#     print(mean_shap.head(10).to_string())

#     return sv, mean_shap


# # ════════════════════════════════════════════════════════
# # 5. 개별 케이스 분석 (선택)
# # ════════════════════════════════════════════════════════
# def analyze_single_case(model, X_val: pd.DataFrame, idx: int = 0, save_dir: str = "."):
#     """특정 환자 1명의 예측 이유 분석"""
#     explainer = shap.TreeExplainer(model)
#     explanation = explainer(X_val.iloc[[idx]])

#     plt.figure()
#     shap.plots.waterfall(explanation[0], show=False)
#     plt.title(f"케이스 #{idx} 예측 분석", fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_case_{idx}.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 케이스 분석 저장: {save_dir}/shap_case_{idx}.png")


# # ════════════════════════════════════════════════════════
# # 메인 실행
# # ════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     TRAIN_PATH = "/Users/admin/nanim/infertility-prediction/train.csv"   # ← 경로 수정
#     TEST_PATH  = "/Users/admin/nanim/infertility-prediction/test.csv"    # ← 없으면 None

#     # 1. 로드
#     train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

#     # 2. 전처리
#     X, y, encoders = preprocess(train_df, is_train=True)

#     # 3. 분할
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )
#     print(f"\n학습: {X_train.shape}, 검증: {X_val.shape}")
#     print(f"성공률 — 학습: {y_train.mean():.3f}, 검증: {y_val.mean():.3f}")

#     # 4. LightGBM 학습
#     model = train_lgb(X_train, y_train, X_val, y_val)

#     # 5. SHAP 분석
#     shap_vals, importance = analyze_shap(model, X_val, save_dir=".")

#     # 6. 개별 케이스 (첫 번째 샘플)
#     analyze_single_case(model, X_val, idx=0, save_dir=".")

#     # 7. 테스트셋 예측 (있을 경우)
#     if test_df is not None:
#         X_test, _, _ = preprocess(test_df, is_train=False, encoders=encoders)
#         test_pred = model.predict_proba(X_test)[:, 1]
#         submission = pd.DataFrame({
#             "ID": test_df["ID"],
#             "임신 성공 여부": test_pred
#         })
#         submission.to_csv("submission.csv", index=False)
#         print("\n✅ submission.csv 저장 완료")



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# #════════════════════════════════════════════════════════
# # Catboost
# #════════════════════════════════════════════════════════

# import pandas as pd
# import numpy as np
# from catboost import CatBoostClassifier, Pool
# import shap
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, classification_report
# import warnings
# warnings.filterwarnings("ignore")

# # ── 한글 폰트 설정 ─────────────────────────────────────
# plt.rcParams["font.family"] = "AppleGothic"   # Mac
# # plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
# plt.rcParams["axes.unicode_minus"] = False


# # ════════════════════════════════════════════════════════
# # 1. 데이터 로드
# # ════════════════════════════════════════════════════════
# def load_data(train_path: str, test_path: str = None):
#     train = pd.read_csv(train_path)
#     test  = pd.read_csv(test_path) if test_path else None
#     print(f"✅ 학습 데이터: {train.shape}")
#     if test is not None:
#         print(f"✅ 테스트 데이터: {test.shape}")
#     return train, test


# # ════════════════════════════════════════════════════════
# # 2. 전처리
# # CatBoost는 범주형을 자체 처리 → Label Encoding 불필요
# # 결측값도 자체 처리 → 중앙값 대체 불필요
# # ════════════════════════════════════════════════════════
# DROP_COLS = [
#     "착상 전 유전 검사 사용 여부",  # 99% 결측
#     "PGD 시술 여부",               # 99% 결측
#     "PGS 시술 여부",               # 99% 결측
#     "난자 해동 경과일",             # 99% 결측
# ]

# AGE_ORDER = {
#     "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
#     "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
# }

# COUNT_COLS = [
#     "총 시술 횟수", "클리닉 내 총 시술 횟수",
#     "IVF 시술 횟수", "DI 시술 횟수",
#     "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
#     "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
# ]

# def parse_count(val):
#     if pd.isna(val): return np.nan
#     val = str(val).strip()
#     if "이상" in val: return int(val[0]) + 1
#     digits = "".join(filter(str.isdigit, val))
#     return int(digits) if digits else np.nan


# def preprocess(df: pd.DataFrame, is_train: bool = True):
#     df = df.copy()

#     # 타깃 분리
#     target = None
#     if is_train and "임신 성공 여부" in df.columns:
#         target = df.pop("임신 성공 여부")

#     df.drop(columns=["ID"], errors="ignore", inplace=True)
#     df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

#     # 나이 순서 인코딩
#     df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

#     # 횟수 컬럼 수치화
#     for col in COUNT_COLS:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_count)

#     # 결측 여부 이진 피처
#     for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
#         if col in df.columns:
#             df[f"{col}_결측"] = df[col].isna().astype(int)

#     # 범주형 컬럼 목록 추출 (CatBoost에 전달용)
#     # 결측값은 빈 문자열로 채워야 CatBoost가 인식
#     cat_cols = df.select_dtypes(include="object").columns.tolist()
#     for col in cat_cols:
#         df[col] = df[col].fillna("unknown")

#     print(f"✅ 전처리 완료: {df.shape} | 범주형 컬럼 수: {len(cat_cols)}")
#     return df, target, cat_cols


# # ════════════════════════════════════════════════════════
# # 3. CatBoost 학습
# # ════════════════════════════════════════════════════════
# def train_cat(X_train, y_train, X_val, y_val, cat_cols):
#     print("\n🐱 CatBoost 학습 중...")

#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#     # Pool 객체: 범주형 컬럼 명시
#     train_pool = Pool(X_train, y_train, cat_features=cat_cols)
#     val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)

#     model = CatBoostClassifier(
#         loss_function="Logloss",
#         eval_metric="AUC",
#         learning_rate=0.05,
#         depth=6,
#         iterations=1000,
#         random_seed=42,
#         scale_pos_weight=pos_weight,
#         early_stopping_rounds=50,
#         verbose=100,
#     )
#     model.fit(train_pool, eval_set=val_pool)

#     pred = model.predict_proba(val_pool)[:, 1]
#     auc  = roc_auc_score(y_val, pred)
#     print(f"\n✅ Validation AUC: {auc:.4f}")
#     print(classification_report(y_val, (pred > 0.5).astype(int),
#                                  target_names=["실패", "성공"]))
#     return model, pred


# # ════════════════════════════════════════════════════════
# # 4. SHAP 분석
# # ════════════════════════════════════════════════════════
# def analyze_shap(model, X_val: pd.DataFrame, cat_cols: list, save_dir: str = "."):
#     print("\n🔍 SHAP 분석 중...")
#     val_pool    = Pool(X_val, cat_features=cat_cols)
#     explainer   = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(val_pool)

#     sv = shap_values[1] if isinstance(shap_values, list) else shap_values

#     # Summary Plot
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, show=False, max_display=20)
#     plt.title("SHAP Summary — 피처별 임신 성공 영향도", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 shap_summary.png 저장")

#     # Bar Plot
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
#     plt.title("SHAP Feature Importance", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 shap_importance.png 저장")

#     mean_shap = pd.Series(
#         np.abs(sv).mean(axis=0), index=X_val.columns
#     ).sort_values(ascending=False)
#     print("\n📋 상위 10개 중요 피처:")
#     print(mean_shap.head(10).to_string())
#     return sv, mean_shap


# def analyze_single_case(model, X_val: pd.DataFrame, cat_cols: list,
#                          idx: int = 0, save_dir: str = "."):
#     """특정 케이스 1개 분석"""
#     val_pool    = Pool(X_val.iloc[[idx]], cat_features=cat_cols)
#     explainer   = shap.TreeExplainer(model)
#     explanation = explainer(X_val.iloc[[idx]])
#     plt.figure()
#     shap.plots.waterfall(explanation[0], show=False)
#     plt.title(f"케이스 #{idx} 예측 분석", fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_case_{idx}.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 shap_case_{idx}.png 저장")


# # ════════════════════════════════════════════════════════
# # 메인 실행
# # ════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     TRAIN_PATH = "data/train.csv"   # ← 경로 수정
#     TEST_PATH  = "data/test.csv"    # ← 없으면 None
#     SAVE_DIR   = "."

#     # 1. 로드
#     train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

#     # 2. 전처리
#     X, y, cat_cols = preprocess(train_df, is_train=True)

#     # 3. 학습/검증 분할
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )
#     print(f"\n학습: {X_train.shape} | 검증: {X_val.shape}")
#     print(f"성공률 — 학습: {y_train.mean():.3f} | 검증: {y_val.mean():.3f}")

#     # 4. CatBoost 학습
#     model, val_pred = train_cat(X_train, y_train, X_val, y_val, cat_cols)

#     # 5. SHAP 분석
#     shap_vals, importance = analyze_shap(model, X_val, cat_cols, save_dir=SAVE_DIR)
#     analyze_single_case(model, X_val, cat_cols, idx=0, save_dir=SAVE_DIR)

#     # 6. 테스트셋 예측 및 제출 파일
#     if test_df is not None:
#         X_test, _, _ = preprocess(test_df, is_train=False)
#         test_pool     = Pool(X_test, cat_features=cat_cols)
#         final_pred    = model.predict_proba(test_pool)[:, 1]
#         submission    = pd.DataFrame({
#             "ID": test_df["ID"],
#             "임신 성공 여부": final_pred,
#         })
#         submission.to_csv("submission.csv", index=False)
#         print("\n✅ submission.csv 저장 완료")

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════
# XGBoost + CatBoost
# ═════════════════════════════════════════════════════

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from catboost import CatBoostClassifier, Pool
# import shap
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score, classification_report
# import warnings
# warnings.filterwarnings("ignore")

# # ── 한글 폰트 설정 ─────────────────────────────────────
# plt.rcParams["font.family"] = "AppleGothic"   # Mac
# # plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
# plt.rcParams["axes.unicode_minus"] = False


# # ════════════════════════════════════════════════════════
# # 1. 데이터 로드
# # ════════════════════════════════════════════════════════
# def load_data(train_path: str, test_path: str = None):
#     train = pd.read_csv(train_path)
#     test  = pd.read_csv(test_path) if test_path else None
#     print(f"✅ 학습 데이터: {train.shape}")
#     if test is not None:
#         print(f"✅ 테스트 데이터: {test.shape}")
#     return train, test


# # ════════════════════════════════════════════════════════
# # 2. 전처리
# # ════════════════════════════════════════════════════════
# DROP_COLS = [
#     "착상 전 유전 검사 사용 여부",
#     "PGD 시술 여부",
#     "PGS 시술 여부",
#     "난자 해동 경과일",
# ]

# AGE_ORDER = {
#     "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
#     "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
# }

# COUNT_COLS = [
#     "총 시술 횟수", "클리닉 내 총 시술 횟수",
#     "IVF 시술 횟수", "DI 시술 횟수",
#     "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
#     "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
# ]

# def parse_count(val):
#     if pd.isna(val): return np.nan
#     val = str(val).strip()
#     if "이상" in val: return int(val[0]) + 1
#     digits = "".join(filter(str.isdigit, val))
#     return int(digits) if digits else np.nan


# def preprocess_cat(df: pd.DataFrame, is_train: bool = True):
#     """CatBoost용 전처리 — 범주형 그대로 유지"""
#     df = df.copy()

#     target = None
#     if is_train and "임신 성공 여부" in df.columns:
#         target = df.pop("임신 성공 여부")

#     df.drop(columns=["ID"], errors="ignore", inplace=True)
#     df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

#     df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

#     for col in COUNT_COLS:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_count)

#     for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
#         if col in df.columns:
#             df[f"{col}_결측"] = df[col].isna().astype(int)

#     # 범주형 결측 → "unknown"
#     cat_cols = df.select_dtypes(include="object").columns.tolist()
#     for col in cat_cols:
#         df[col] = df[col].fillna("unknown")

#     return df, target, cat_cols


# def preprocess_xgb(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
#     """XGBoost용 전처리 — Label Encoding + 결측 중앙값 대체"""
#     df = df.copy()

#     target = None
#     if is_train and "임신 성공 여부" in df.columns:
#         target = df.pop("임신 성공 여부")

#     df.drop(columns=["ID"], errors="ignore", inplace=True)
#     df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

#     df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

#     for col in COUNT_COLS:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_count)

#     for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
#         if col in df.columns:
#             df[f"{col}_결측"] = df[col].isna().astype(int)

#     # Label Encoding
#     cat_cols = df.select_dtypes(include="object").columns.tolist()
#     if encoders is None:
#         encoders = {}

#     for col in cat_cols:
#         if is_train:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             encoders[col] = le
#         else:
#             le = encoders.get(col)
#             if le:
#                 df[col] = df[col].astype(str).apply(
#                     lambda x: le.transform([x])[0] if x in le.classes_ else -1
#                 )

#     # 수치형 결측 중앙값 대체
#     num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
#     if is_train:
#         medians = df[num_cols].median()
#         encoders["medians"] = medians
#     else:
#         medians = encoders.get("medians", df[num_cols].median())
#     df[num_cols] = df[num_cols].fillna(medians)

#     return df, target, encoders


# # ════════════════════════════════════════════════════════
# # 3. 모델 학습
# # ════════════════════════════════════════════════════════
# def train_xgb(X_train, y_train, X_val, y_val):
#     print("\n⚡ XGBoost 학습 중...")
#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
#     model = xgb.XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="auc",
#         learning_rate=0.05,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=pos_weight,
#         n_estimators=1000,
#         random_state=42,
#         verbosity=0,
#         early_stopping_rounds=50,
#     )
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=100,
#     )
#     pred = model.predict_proba(X_val)[:, 1]
#     print(f"  ➜ XGB AUC: {roc_auc_score(y_val, pred):.4f}")
#     return model, pred


# def train_cat(X_train, y_train, X_val, y_val, cat_cols):
#     print("\n🐱 CatBoost 학습 중...")
#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#     train_pool = Pool(X_train, y_train, cat_features=cat_cols)
#     val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)

#     model = CatBoostClassifier(
#         loss_function="Logloss",
#         eval_metric="AUC",
#         learning_rate=0.05,
#         depth=6,
#         iterations=1000,
#         random_seed=42,
#         scale_pos_weight=pos_weight,
#         early_stopping_rounds=50,
#         verbose=100,
#     )
#     model.fit(train_pool, eval_set=val_pool)

#     pred = model.predict_proba(val_pool)[:, 1]
#     print(f"  ➜ CAT AUC: {roc_auc_score(y_val, pred):.4f}")
#     return model, pred


# # ════════════════════════════════════════════════════════
# # 4. 앙상블
# # ════════════════════════════════════════════════════════
# def find_best_weights(preds: dict, y_val, step: float = 0.05):
#     """그리드 서치로 최적 가중치 탐색"""
#     print("\n🔎 최적 가중치 탐색 중...")
#     best_auc, best_w = 0, None
#     keys = list(preds.keys())
#     ws   = np.arange(0, 1 + step, step)

#     for w0 in ws:
#         w1 = round(1 - w0, 2)
#         if w1 < 0: continue
#         w       = {keys[0]: w0, keys[1]: w1}
#         blended = sum(preds[k] * w[k] for k in keys)
#         auc     = roc_auc_score(y_val, blended)
#         if auc > best_auc:
#             best_auc, best_w = auc, w

#     print(f"  최적 AUC: {best_auc:.4f} | 최적 가중치: {best_w}")
#     return best_w


# def ensemble(preds: dict, y_val, weights: dict = None):
#     if weights is None:
#         weights = {k: 1 / len(preds) for k in preds}
#     blended = sum(preds[k] * weights[k] for k in preds)
#     auc = roc_auc_score(y_val, blended)
#     print(f"\n🏆 앙상블 AUC: {auc:.4f}  (가중치: {weights})")
#     print(classification_report(y_val, (blended > 0.5).astype(int),
#                                  target_names=["실패", "성공"]))
#     return blended


# # ════════════════════════════════════════════════════════
# # 5. SHAP 분석 (CatBoost 기준)
# # ════════════════════════════════════════════════════════
# def analyze_shap(model, X_val: pd.DataFrame, cat_cols: list, save_dir: str = "."):
#     print("\n🔍 SHAP 분석 중 (CatBoost 기준)...")
#     val_pool    = Pool(X_val, cat_features=cat_cols)
#     explainer   = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(val_pool)
#     sv = shap_values[1] if isinstance(shap_values, list) else shap_values

#     # Summary Plot
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, show=False, max_display=20)
#     plt.title("SHAP Summary — 피처별 임신 성공 영향도", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print("  📊 shap_summary.png 저장")

#     # Bar Plot
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
#     plt.title("SHAP Feature Importance", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print("  📊 shap_importance.png 저장")

#     mean_shap = pd.Series(
#         np.abs(sv).mean(axis=0), index=X_val.columns
#     ).sort_values(ascending=False)
#     print("\n📋 상위 10개 중요 피처:")
#     print(mean_shap.head(10).to_string())
#     return sv, mean_shap


# def analyze_single_case(model, X_val: pd.DataFrame, cat_cols: list,
#                          idx: int = 0, save_dir: str = "."):
#     explainer   = shap.TreeExplainer(model)
#     explanation = explainer(X_val.iloc[[idx]])
#     plt.figure()
#     shap.plots.waterfall(explanation[0], show=False)
#     plt.title(f"케이스 #{idx} 예측 분석", fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_case_{idx}.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 shap_case_{idx}.png 저장")


# # ════════════════════════════════════════════════════════
# # 메인 실행
# # ════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     TRAIN_PATH = "data/train.csv"   # ← 경로 수정
#     TEST_PATH  = "data/test.csv"    # ← 없으면 None
#     SAVE_DIR   = "."

#     # 1. 로드
#     train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

#     # 2. 전처리 (모델별로 따로)
#     X_cat, y, cat_cols        = preprocess_cat(train_df, is_train=True)
#     X_xgb, _, xgb_encoders   = preprocess_xgb(train_df, is_train=True)

#     # 3. 동일한 인덱스로 분할 (두 모델이 같은 샘플 보도록)
#     from sklearn.model_selection import train_test_split
#     idx_train, idx_val = train_test_split(
#         np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
#     )

#     X_cat_train, X_cat_val = X_cat.iloc[idx_train], X_cat.iloc[idx_val]
#     X_xgb_train, X_xgb_val = X_xgb.iloc[idx_train], X_xgb.iloc[idx_val]
#     y_train, y_val          = y.iloc[idx_train],     y.iloc[idx_val]

#     print(f"\n학습: {X_cat_train.shape} | 검증: {X_cat_val.shape}")
#     print(f"성공률 — 학습: {y_train.mean():.3f} | 검증: {y_val.mean():.3f}")

#     # 4. 모델 학습
#     xgb_model, xgb_pred = train_xgb(X_xgb_train, y_train, X_xgb_val, y_val)
#     cat_model, cat_pred = train_cat(X_cat_train, y_train, X_cat_val, y_val, cat_cols)

#     # 5. 앙상블
#     preds  = {"xgb": xgb_pred, "cat": cat_pred}
#     best_w = find_best_weights(preds, y_val, step=0.05)
#     blended = ensemble(preds, y_val, weights=best_w)

#     # 6. SHAP 분석 (CatBoost 기준)
#     shap_vals, importance = analyze_shap(cat_model, X_cat_val, cat_cols, save_dir=SAVE_DIR)
#     analyze_single_case(cat_model, X_cat_val, cat_cols, idx=0, save_dir=SAVE_DIR)

#     # 7. 테스트셋 예측 및 제출 파일
#     if test_df is not None:
#         X_cat_test, _, _  = preprocess_cat(test_df, is_train=False)
#         X_xgb_test, _, _  = preprocess_xgb(test_df, is_train=False, encoders=xgb_encoders)

#         test_preds = {
#             "xgb": xgb_model.predict_proba(X_xgb_test)[:, 1],
#             "cat": cat_model.predict_proba(Pool(X_cat_test, cat_features=cat_cols))[:, 1],
#         }
#         final_pred = sum(test_preds[k] * best_w[k] for k in test_preds)

#         submission = pd.DataFrame({
#             "ID": test_df["ID"],
#             "임신 성공 여부": final_pred,
#         })
#         submission.to_csv("submission.csv", index=False)
#         print("\n✅ xgbcatsubmission.csv 저장 완료")

# ───────────────────────────────────────

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from catboost import CatBoostClassifier, Pool
# import shap
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score, classification_report
# import warnings
# warnings.filterwarnings("ignore")

# # ── 한글 폰트 설정 ─────────────────────────────────────
# plt.rcParams["font.family"] = "AppleGothic"   # Mac
# # plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
# plt.rcParams["axes.unicode_minus"] = False


# # ════════════════════════════════════════════════════════
# # 1. 데이터 로드
# # ════════════════════════════════════════════════════════
# def load_data(train_path: str, test_path: str = None):
#     train = pd.read_csv(train_path)
#     test  = pd.read_csv(test_path) if test_path else None
#     print(f"✅ 학습 데이터: {train.shape}")
#     if test is not None:
#         print(f"✅ 테스트 데이터: {test.shape}")
#     return train, test


# # ════════════════════════════════════════════════════════
# # 2. 피처 엔지니어링
# # SHAP 상위 피처 기반 파생 변수 생성
# # ※ 횟수 수치화 / 나이 인코딩 이후에 호출
# # ════════════════════════════════════════════════════════
# def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

#     # ── 배아 효율 ──────────────────────────────────────
#     # 이식된 배아 수 / 총 생성 배아 수 → 배아 이식 효율
#     df["배아_이식률"] = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)

#     # 저장된 배아 수 / 총 생성 배아 수 → 여유 배아 비율
#     df["배아_저장률"] = df["저장된 배아 수"] / (df["총 생성 배아 수"] + 1)

#     # 미세주입 배아 이식 수 / 이식된 배아 수 → ICSI 이식 비율
#     df["ICSI_이식비율"] = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)

#     # ── 난자 품질 ──────────────────────────────────────
#     # 미세주입 생성 배아 / 미세주입된 난자 수 → 수정 성공률
#     df["수정_성공률"] = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + 1)

#     # 혼합된 난자 / 수집된 신선 난자 → 난자 활용률
#     df["난자_활용률"] = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)

#     # ── 나이 × 배아 교호작용 ───────────────────────────
#     # 나이가 많을수록 배아 수의 중요도가 더 커짐
#     df["나이_x_배아수"] = df["시술 당시 나이"] * df["이식된 배아 수"]
#     df["나이_x_생성배아"] = df["시술 당시 나이"] * df["총 생성 배아 수"]

#     # ── 시술 경험 ──────────────────────────────────────
#     # 총 임신 횟수 / 총 시술 횟수 → 과거 임신 성공률
#     df["과거_임신율"] = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)

#     # 총 출산 횟수 / 총 임신 횟수 → 임신→출산 성공률
#     df["임신_출산율"] = df["총 출산 횟수"] / (df["총 임신 횟수"] + 1)

#     # ── 타이밍 ─────────────────────────────────────────
#     # 난자 혼합 ~ 배아 이식 경과일 차이 → 배양 기간
#     df["배양_기간"] = df["배아 이식 경과일"] - df["난자 혼합 경과일"]

#     # 배아 이식 경과일이 짧을수록 신선 이식 가능성
#     df["이식_빠름"] = (df["배아 이식 경과일"] <= 3).astype(int)

#     print(f"  ✅ 파생 피처 11개 추가 → 총 {df.shape[1]}개 컬럼")
#     return df


# # ════════════════════════════════════════════════════════
# # 3. 전처리
# # ════════════════════════════════════════════════════════
# DROP_COLS = [
#     "착상 전 유전 검사 사용 여부",
#     "PGD 시술 여부",
#     "PGS 시술 여부",
#     "난자 해동 경과일",
# ]

# AGE_ORDER = {
#     "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
#     "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
# }

# COUNT_COLS = [
#     "총 시술 횟수", "클리닉 내 총 시술 횟수",
#     "IVF 시술 횟수", "DI 시술 횟수",
#     "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
#     "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
# ]

# def parse_count(val):
#     if pd.isna(val): return np.nan
#     val = str(val).strip()
#     if "이상" in val: return int(val[0]) + 1
#     digits = "".join(filter(str.isdigit, val))
#     return int(digits) if digits else np.nan


# def preprocess_cat(df: pd.DataFrame, is_train: bool = True):
#     """CatBoost용 전처리 — 범주형 그대로 유지"""
#     df = df.copy()

#     target = None
#     if is_train and "임신 성공 여부" in df.columns:
#         target = df.pop("임신 성공 여부")

#     df.drop(columns=["ID"], errors="ignore", inplace=True)
#     df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

#     df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

#     for col in COUNT_COLS:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_count)

#     for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
#         if col in df.columns:
#             df[f"{col}_결측"] = df[col].isna().astype(int)

#     # 피처 엔지니어링
#     df = feature_engineering(df)

#     # 범주형 결측 → "unknown"
#     cat_cols = df.select_dtypes(include="object").columns.tolist()
#     for col in cat_cols:
#         df[col] = df[col].fillna("unknown")

#     print(f"✅ CatBoost 전처리 완료: {df.shape} | 범주형: {len(cat_cols)}개")
#     return df, target, cat_cols


# def preprocess_xgb(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
#     """XGBoost용 전처리 — Label Encoding + 결측 중앙값 대체"""
#     df = df.copy()

#     target = None
#     if is_train and "임신 성공 여부" in df.columns:
#         target = df.pop("임신 성공 여부")

#     df.drop(columns=["ID"], errors="ignore", inplace=True)
#     df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

#     df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)

#     for col in COUNT_COLS:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_count)

#     for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
#         if col in df.columns:
#             df[f"{col}_결측"] = df[col].isna().astype(int)

#     # 피처 엔지니어링
#     df = feature_engineering(df)

#     # Label Encoding
#     cat_cols = df.select_dtypes(include="object").columns.tolist()
#     if encoders is None:
#         encoders = {}

#     for col in cat_cols:
#         if is_train:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             encoders[col] = le
#         else:
#             le = encoders.get(col)
#             if le:
#                 df[col] = df[col].astype(str).apply(
#                     lambda x: le.transform([x])[0] if x in le.classes_ else -1
#                 )

#     # 수치형 결측 중앙값 대체
#     num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
#     if is_train:
#         medians = df[num_cols].median()
#         encoders["medians"] = medians
#     else:
#         medians = encoders.get("medians", df[num_cols].median())
#     df[num_cols] = df[num_cols].fillna(medians)

#     print(f"✅ XGBoost 전처리 완료: {df.shape}")
#     return df, target, encoders


# # ════════════════════════════════════════════════════════
# # 4. 모델 학습
# # ════════════════════════════════════════════════════════
# def train_xgb(X_train, y_train, X_val, y_val):
#     print("\n⚡ XGBoost 학습 중...")
#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
#     model = xgb.XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="auc",
#         learning_rate=0.05,
#         max_depth=6,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=pos_weight,
#         n_estimators=1000,
#         random_state=42,
#         verbosity=0,
#         early_stopping_rounds=50,
#     )
#     model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
#     pred = model.predict_proba(X_val)[:, 1]
#     print(f"  ➜ XGB AUC: {roc_auc_score(y_val, pred):.4f}")
#     return model, pred


# def train_cat(X_train, y_train, X_val, y_val, cat_cols):
#     print("\n🐱 CatBoost 학습 중...")
#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#     train_pool = Pool(X_train, y_train, cat_features=cat_cols)
#     val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)

#     model = CatBoostClassifier(
#         loss_function="Logloss",
#         eval_metric="AUC",
#         learning_rate=0.05,
#         depth=6,
#         iterations=1000,
#         random_seed=42,
#         scale_pos_weight=pos_weight,
#         early_stopping_rounds=50,
#         verbose=100,
#     )
#     model.fit(train_pool, eval_set=val_pool)
#     pred = model.predict_proba(val_pool)[:, 1]
#     print(f"  ➜ CAT AUC: {roc_auc_score(y_val, pred):.4f}")
#     return model, pred


# # ════════════════════════════════════════════════════════
# # 5. 앙상블
# # ════════════════════════════════════════════════════════
# def find_best_weights(preds: dict, y_val, step: float = 0.05):
#     print("\n🔎 최적 가중치 탐색 중...")
#     best_auc, best_w = 0, None
#     keys = list(preds.keys())
#     ws   = np.arange(0, 1 + step, step)

#     for w0 in ws:
#         w1 = round(1 - w0, 2)
#         if w1 < 0: continue
#         w       = {keys[0]: w0, keys[1]: w1}
#         blended = sum(preds[k] * w[k] for k in keys)
#         auc     = roc_auc_score(y_val, blended)
#         if auc > best_auc:
#             best_auc, best_w = auc, w

#     print(f"  최적 AUC: {best_auc:.4f} | 최적 가중치: {best_w}")
#     return best_w


# def ensemble(preds: dict, y_val, weights: dict = None):
#     if weights is None:
#         weights = {k: 1 / len(preds) for k in preds}
#     blended = sum(preds[k] * weights[k] for k in preds)
#     auc = roc_auc_score(y_val, blended)
#     print(f"\n🏆 앙상블 AUC: {auc:.4f}  (가중치: {weights})")
#     print(classification_report(y_val, (blended > 0.5).astype(int),
#                                  target_names=["실패", "성공"]))
#     return blended


# # ════════════════════════════════════════════════════════
# # 6. SHAP 분석 (CatBoost 기준)
# # ════════════════════════════════════════════════════════
# def analyze_shap(model, X_val: pd.DataFrame, cat_cols: list, save_dir: str = "."):
#     print("\n🔍 SHAP 분석 중 (CatBoost 기준)...")
#     val_pool    = Pool(X_val, cat_features=cat_cols)
#     explainer   = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(val_pool)
#     sv = shap_values[1] if isinstance(shap_values, list) else shap_values

#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, show=False, max_display=20)
#     plt.title("SHAP Summary — 피처별 임신 성공 영향도", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print("  📊 shap_summary.png 저장")

#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
#     plt.title("SHAP Feature Importance", fontsize=13)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print("  📊 shap_importance.png 저장")

#     mean_shap = pd.Series(
#         np.abs(sv).mean(axis=0), index=X_val.columns
#     ).sort_values(ascending=False)
#     print("\n📋 상위 15개 중요 피처:")
#     print(mean_shap.head(15).to_string())
#     return sv, mean_shap


# def analyze_single_case(model, X_val: pd.DataFrame, cat_cols: list,
#                          idx: int = 0, save_dir: str = "."):
#     explainer   = shap.TreeExplainer(model)
#     explanation = explainer(X_val.iloc[[idx]])
#     plt.figure()
#     shap.plots.waterfall(explanation[0], show=False)
#     plt.title(f"케이스 #{idx} 예측 분석", fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/shap_case_{idx}.png", dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  📊 shap_case_{idx}.png 저장")


# # ════════════════════════════════════════════════════════
# # 메인 실행
# # ════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     TRAIN_PATH = "data/train.csv"   # ← 경로 수정
#     TEST_PATH  = "data/test.csv"    # ← 없으면 None
#     SAVE_DIR   = "."

#     # 1. 로드
#     train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

#     # 2. 전처리 (모델별)
#     X_cat, y, cat_cols      = preprocess_cat(train_df, is_train=True)
#     X_xgb, _, xgb_encoders  = preprocess_xgb(train_df, is_train=True)

#     # 3. 동일 인덱스로 분할
#     idx_train, idx_val = train_test_split(
#         np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
#     )
#     X_cat_train, X_cat_val = X_cat.iloc[idx_train], X_cat.iloc[idx_val]
#     X_xgb_train, X_xgb_val = X_xgb.iloc[idx_train], X_xgb.iloc[idx_val]
#     y_train, y_val          = y.iloc[idx_train],     y.iloc[idx_val]

#     print(f"\n학습: {X_cat_train.shape} | 검증: {X_cat_val.shape}")
#     print(f"성공률 — 학습: {y_train.mean():.3f} | 검증: {y_val.mean():.3f}")

#     # 4. 모델 학습
#     xgb_model, xgb_pred = train_xgb(X_xgb_train, y_train, X_xgb_val, y_val)
#     cat_model, cat_pred = train_cat(X_cat_train, y_train, X_cat_val, y_val, cat_cols)

#     # 5. 앙상블
#     preds   = {"xgb": xgb_pred, "cat": cat_pred}
#     best_w  = find_best_weights(preds, y_val, step=0.05)
#     blended = ensemble(preds, y_val, weights=best_w)

#     # 6. SHAP 분석
#     shap_vals, importance = analyze_shap(cat_model, X_cat_val, cat_cols, save_dir=SAVE_DIR)
#     analyze_single_case(cat_model, X_cat_val, cat_cols, idx=0, save_dir=SAVE_DIR)

#     # 7. 테스트셋 예측 및 제출 파일
#     if test_df is not None:
#         X_cat_test, _, _  = preprocess_cat(test_df, is_train=False)
#         X_xgb_test, _, _  = preprocess_xgb(test_df, is_train=False, encoders=xgb_encoders)

#         test_preds = {
#             "xgb": xgb_model.predict_proba(X_xgb_test)[:, 1],
#             "cat": cat_model.predict_proba(Pool(X_cat_test, cat_features=cat_cols))[:, 1],
#         }
#         final_pred = sum(test_preds[k] * best_w[k] for k in test_preds)

#         submission = pd.DataFrame({
#             "ID": test_df["ID"],
#             "임신 성공 여부": final_pred,
#         })
#         submission.to_csv("submission.csv", index=False)
#         print("\n✅ submission.csv 저장 완료")



# ════════════════════════════════════════════════════════

#XGBoost + CatBoost → Optuna 튜닝 → 가중 평균 앙상블 → SHAP 해석

# ════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import shap
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna 로그 간소화

# ── 한글 폰트 설정 ─────────────────────────────────────
plt.rcParams["font.family"] = "AppleGothic"   # Mac
# plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════
def load_data(train_path: str, test_path: str = None):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path) if test_path else None
    print(f"✅ 학습 데이터: {train.shape}")
    if test is not None:
        print(f"✅ 테스트 데이터: {test.shape}")
    return train, test


# ════════════════════════════════════════════════════════
# 2. 피처 엔지니어링
# ════════════════════════════════════════════════════════
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["배아_이식률"]    = df["이식된 배아 수"] / (df["총 생성 배아 수"] + 1)
    df["배아_저장률"]    = df["저장된 배아 수"] / (df["총 생성 배아 수"] + 1)
    df["ICSI_이식비율"]  = df["미세주입 배아 이식 수"] / (df["이식된 배아 수"] + 1)
    df["수정_성공률"]    = df["미세주입에서 생성된 배아 수"] / (df["미세주입된 난자 수"] + 1)
    df["난자_활용률"]    = df["혼합된 난자 수"] / (df["수집된 신선 난자 수"] + 1)
    df["나이_x_배아수"]  = df["시술 당시 나이"] * df["이식된 배아 수"]
    df["나이_x_생성배아"] = df["시술 당시 나이"] * df["총 생성 배아 수"]
    df["과거_임신율"]    = df["총 임신 횟수"] / (df["총 시술 횟수"] + 1)
    df["임신_출산율"]    = df["총 출산 횟수"] / (df["총 임신 횟수"] + 1)
    df["배양_기간"]      = df["배아 이식 경과일"] - df["난자 혼합 경과일"]
    df["이식_빠름"]      = (df["배아 이식 경과일"] <= 3).astype(int)
    return df


# ════════════════════════════════════════════════════════
# 3. 전처리
# ════════════════════════════════════════════════════════
DROP_COLS = [
    "착상 전 유전 검사 사용 여부", "PGD 시술 여부",
    "PGS 시술 여부", "난자 해동 경과일",
]
AGE_ORDER = {
    "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
    "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
}
COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수",  "IVF 임신 횟수",  "DI 임신 횟수",
    "총 출산 횟수",  "IVF 출산 횟수",  "DI 출산 횟수",
]

def parse_count(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if "이상" in val: return int(val[0]) + 1
    digits = "".join(filter(str.isdigit, val))
    return int(digits) if digits else np.nan

def _base_preprocess(df, is_train):
    """공통 전처리 (나이 인코딩, 횟수 수치화, 피처 엔지니어링)"""
    target = None
    if is_train and "임신 성공 여부" in df.columns:
        target = df.pop("임신 성공 여부")
    df.drop(columns=["ID"], errors="ignore", inplace=True)
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)
    df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_ORDER)
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_count)
    for col in ["임신 시도 또는 마지막 임신 경과 연수"]:
        if col in df.columns:
            df[f"{col}_결측"] = df[col].isna().astype(int)
    df = feature_engineering(df)
    return df, target

def preprocess_cat(df: pd.DataFrame, is_train: bool = True):
    df = df.copy()
    df, target = _base_preprocess(df, is_train)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
    print(f"✅ CatBoost 전처리 완료: {df.shape}")
    return df, target, cat_cols

def preprocess_xgb(df: pd.DataFrame, is_train: bool = True, encoders: dict = None):
    df = df.copy()
    df, target = _base_preprocess(df, is_train)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if is_train:
        medians = df[num_cols].median()
        encoders["medians"] = medians
    else:
        medians = encoders.get("medians", df[num_cols].median())
    df[num_cols] = df[num_cols].fillna(medians)
    print(f"✅ XGBoost 전처리 완료: {df.shape}")
    return df, target, encoders


# ════════════════════════════════════════════════════════
# 4. Optuna 튜닝
# ════════════════════════════════════════════════════════
def tune_xgb(X, y, n_trials: int = 50):
    """XGBoost Optuna 튜닝 — 3-Fold CV AUC 최대화"""
    print(f"\n🔬 XGBoost Optuna 튜닝 ({n_trials} trials)...")
    pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "verbosity":        0,
            "scale_pos_weight": pos_weight,
            "n_estimators":     1000,
            "random_state":     42,
            "early_stopping_rounds": 50,
            # ── 튜닝 대상 파라미터 ──
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0, 5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], pred))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  ➜ XGB 최적 AUC: {study.best_value:.4f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


def tune_cat(X, y, cat_cols, n_trials: int = 50):
    """CatBoost Optuna 튜닝 — 3-Fold CV AUC 최대화"""
    print(f"\n🔬 CatBoost Optuna 튜닝 ({n_trials} trials)...")
    pos_weight = (y == 0).sum() / (y == 1).sum()

    def objective(trial):
        params = {
            "loss_function":      "Logloss",
            "eval_metric":        "AUC",
            "random_seed":        42,
            "verbose":            False,
            "scale_pos_weight":   pos_weight,
            "early_stopping_rounds": 50,
            # ── 튜닝 대상 파라미터 ──
            "iterations":         trial.suggest_int("iterations", 300, 1000),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":              trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength":    trial.suggest_float("random_strength", 0, 2),
            "border_count":       trial.suggest_int("border_count", 32, 255),
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, val_idx in cv.split(X, y):
            train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
            val_pool   = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_cols)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool)
            pred = model.predict_proba(val_pool)[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], pred))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  ➜ CAT 최적 AUC: {study.best_value:.4f}")
    print(f"  ➜ 최적 파라미터: {study.best_params}")
    return study.best_params


# ════════════════════════════════════════════════════════
# 5. 최적 파라미터로 최종 모델 학습
# ════════════════════════════════════════════════════════
def train_xgb(X_train, y_train, X_val, y_val, best_params: dict = None):
    print("\n⚡ XGBoost 최종 학습 중...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "scale_pos_weight": pos_weight, "n_estimators": 1000,
        "random_state": 42, "verbosity": 0, "early_stopping_rounds": 50,
        **(best_params or {
            "learning_rate": 0.05, "max_depth": 6,
            "subsample": 0.8,      "colsample_bytree": 0.8,
        }),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    pred = model.predict_proba(X_val)[:, 1]
    print(f"  ➜ XGB AUC: {roc_auc_score(y_val, pred):.4f}")
    return model, pred


def train_cat(X_train, y_train, X_val, y_val, cat_cols, best_params: dict = None):
    print("\n🐱 CatBoost 최종 학습 중...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "loss_function": "Logloss", "eval_metric": "AUC",
        "random_seed": 42, "scale_pos_weight": pos_weight,
        "early_stopping_rounds": 50, "verbose": 100,
        **(best_params or {
            "learning_rate": 0.05, "depth": 6, "iterations": 1000,
        }),
    }
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    pred = model.predict_proba(val_pool)[:, 1]
    print(f"  ➜ CAT AUC: {roc_auc_score(y_val, pred):.4f}")
    return model, pred


# ════════════════════════════════════════════════════════
# 6. 앙상블
# ════════════════════════════════════════════════════════
def find_best_weights(preds: dict, y_val, step: float = 0.05):
    print("\n🔎 최적 가중치 탐색 중...")
    best_auc, best_w = 0, None
    keys = list(preds.keys())
    for w0 in np.arange(0, 1 + step, step):
        w1 = round(1 - w0, 2)
        if w1 < 0: continue
        w       = {keys[0]: w0, keys[1]: w1}
        blended = sum(preds[k] * w[k] for k in keys)
        auc     = roc_auc_score(y_val, blended)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"  최적 AUC: {best_auc:.4f} | 최적 가중치: {best_w}")
    return best_w

def ensemble(preds: dict, y_val, weights: dict = None):
    if weights is None:
        weights = {k: 1 / len(preds) for k in preds}
    blended = sum(preds[k] * weights[k] for k in preds)
    auc = roc_auc_score(y_val, blended)
    print(f"\n🏆 앙상블 AUC: {auc:.4f}  (가중치: {weights})")
    print(classification_report(y_val, (blended > 0.5).astype(int),
                                 target_names=["실패", "성공"]))
    return blended


# ════════════════════════════════════════════════════════
# 7. SHAP 분석
# ════════════════════════════════════════════════════════
def analyze_shap(model, X_val: pd.DataFrame, cat_cols: list, save_dir: str = "."):
    print("\n🔍 SHAP 분석 중 (CatBoost 기준)...")
    val_pool    = Pool(X_val, cat_features=cat_cols)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(val_pool)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, show=False, max_display=20)
    plt.title("SHAP Summary — 피처별 임신 성공 영향도", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_shap = pd.Series(
        np.abs(sv).mean(axis=0), index=X_val.columns
    ).sort_values(ascending=False)
    print("\n📋 상위 15개 중요 피처:")
    print(mean_shap.head(15).to_string())
    return sv, mean_shap


# ════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    TRAIN_PATH  = "data/train.csv"   # ← 경로 수정
    TEST_PATH   = "data/test.csv"    # ← 없으면 None
    SAVE_DIR    = "."
    N_TRIALS    = 50   # ← 시간 여유 있으면 100으로 올리기 (더 정확)

    # 1. 로드
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # 2. 전처리
    X_cat, y, cat_cols     = preprocess_cat(train_df, is_train=True)
    X_xgb, _, xgb_encoders = preprocess_xgb(train_df, is_train=True)

    # 3. 분할
    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    X_cat_train, X_cat_val = X_cat.iloc[idx_train], X_cat.iloc[idx_val]
    X_xgb_train, X_xgb_val = X_xgb.iloc[idx_train], X_xgb.iloc[idx_val]
    y_train, y_val          = y.iloc[idx_train],     y.iloc[idx_val]

    print(f"\n학습: {X_cat_train.shape} | 검증: {X_cat_val.shape}")

    # 4. Optuna 튜닝 (전체 학습 데이터로 CV)
    xgb_best = tune_xgb(X_xgb.iloc[idx_train], y.iloc[idx_train], n_trials=N_TRIALS)
    cat_best  = tune_cat(X_cat.iloc[idx_train], y.iloc[idx_train], cat_cols, n_trials=N_TRIALS)

    # 5. 최적 파라미터로 최종 학습
    xgb_model, xgb_pred = train_xgb(X_xgb_train, y_train, X_xgb_val, y_val, xgb_best)
    cat_model, cat_pred = train_cat(X_cat_train, y_train, X_cat_val, y_val, cat_cols, cat_best)

    # 6. 앙상블
    preds   = {"xgb": xgb_pred, "cat": cat_pred}
    best_w  = find_best_weights(preds, y_val, step=0.05)
    blended = ensemble(preds, y_val, weights=best_w)

    # 7. SHAP 분석
    analyze_shap(cat_model, X_cat_val, cat_cols, save_dir=SAVE_DIR)

    # 8. 테스트셋 예측 및 제출
    if test_df is not None:
        X_cat_test, _, _  = preprocess_cat(test_df, is_train=False)
        X_xgb_test, _, _  = preprocess_xgb(test_df, is_train=False, encoders=xgb_encoders)
        test_preds = {
            "xgb": xgb_model.predict_proba(X_xgb_test)[:, 1],
            "cat": cat_model.predict_proba(Pool(X_cat_test, cat_features=cat_cols))[:, 1],
        }
        final_pred = sum(test_preds[k] * best_w[k] for k in test_preds)
        submission = pd.DataFrame({"ID": test_df["ID"], "임신 성공 여부": final_pred})
        submission.to_csv("submission.csv", index=False)
        print("\n✅ submission.csv 저장 완료")

