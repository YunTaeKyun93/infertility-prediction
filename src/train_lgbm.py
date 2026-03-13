import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
  train = pd.read_csv(DATA_DIR / 'train.csv')
  test = pd.read_csv(DATA_DIR / 'test.csv')
  return train, test

def preprocess(df):
  df = df.copy()

  if 'ID' in df.columns:
    df = df.drop(columns=["ID"])

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
