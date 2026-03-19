import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 앙상블 (failure 80% + MLP 20%) ──────────────────────
failure = pd.read_csv('submission_failure.csv')
mlp     = pd.read_csv('submission_mlp.csv')

blend = failure.copy()
blend['probability'] = 0.80 * failure.iloc[:,1].values + 0.20 * mlp.iloc[:,1].values
blend.to_csv('submission_blend_f80_m20.csv', index=False)

print("=== 앙상블 완료 ===")
print(f"failure mean:  {failure.iloc[:,1].mean():.4f}")
print(f"mlp     mean:  {mlp.iloc[:,1].mean():.4f}")
print(f"blend   mean:  {blend.iloc[:,1].mean():.4f}")
print(f"상관관계: {np.corrcoef(failure.iloc[:,1].values, mlp.iloc[:,1].values)[0,1]:.4f}")
print("→ submission_blend_f80_m20.csv 저장 완료\n")

# ── 2. SHAP 분석 ────────────────────────────────────────────
print("=== SHAP 분석 시작 ===")
train = pd.read_csv('data/train.csv')

# 간단 전처리
TARGET = '임신 성공 여부'
DROP   = ['ID', TARGET]
X = train.drop(columns=DROP, errors='ignore')
y = train[TARGET]

# 범주형 → 수치형
for col in X.select_dtypes(include='object').columns:
    X[col] = pd.Categorical(X[col]).codes

X = X.fillna(-999)

# LGB 빠르게 학습
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    random_state=42,
    verbose=-1
)
model.fit(X, y)
print("LGB 학습 완료")

# SHAP 계산 (샘플 5000개로 속도 향상)
sample_idx = np.random.choice(len(X), 5000, replace=False)
X_sample   = X.iloc[sample_idx]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# 이진분류: shap_values[1] = 성공(1) 클래스
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# ── 3. SHAP 시각화 ───────────────────────────────────────────

# 그래프 1: Summary Plot (상위 20개)
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_sample, max_display=20, show=False)
plt.title('SHAP Summary Plot — 피처 중요도 TOP 20', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ shap_summary.png 저장")

# 그래프 2: Bar Plot (평균 절댓값)
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_sample, plot_type='bar', max_display=20, show=False)
plt.title('SHAP Feature Importance (평균 |SHAP값|)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("→ shap_bar.png 저장")

# 그래프 3: 상위 5개 피처 이름 출력
mean_shap = np.abs(sv).mean(axis=0)
feat_imp  = pd.DataFrame({'feature': X.columns, 'shap': mean_shap})
feat_imp  = feat_imp.sort_values('shap', ascending=False)

print("\n=== SHAP TOP 10 피처 ===")
print(feat_imp.head(10).to_string(index=False))

print("\n✅ 모두 완료!")
print("제출 파일: submission_blend_f80_m20.csv")
print("SHAP 그래프: shap_summary.png / shap_bar.png")