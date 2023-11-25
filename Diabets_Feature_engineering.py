#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
#import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv(r"datasets/diabetes.csv")

# Genel Bakış
df.head()
df.info()
df.describe()

def grab_col_names(dataframe,cat_th = 10, car_th = 20 ):
  cat_cols = [col for col in df.columns if df[col].dtypes == "O" ]
  num_but_cat = [col for col in df.columns if (df[col].dtypes != "O") and (df[col].nunique() < cat_th)]
  cat_but_car = [col for col in df.columns if (df[col].dtypes == "O") and (df[col].nunique() > car_th)]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car ]
  # Num belirleme
  num_cols = [col for col in df.columns if df[col].dtypes in ["float","int"]]
  num_cols = [col for col in num_cols if col not in num_but_cat ]
  return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(cat_cols)

df.groupby(cat_cols).agg(["describe"])

for cat_col in cat_cols:
    print(df.groupby(cat_col).agg({"Outcome": "mean"}))
print("#################################################")
# Hedef değişkene göre numerik değişkenlerin ortalaması
for num_col in num_cols:
    print(df.groupby("Outcome").agg({num_col: "mean"}))

def outlier_thresholds(dataframe,col_name, q1 = 0.25, q3 = 0.75):
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quartile3 - quartile1
  up_limit = quartile3 + 1.5 * interquantile_range
  low_limit = quartile1 - 1.5 * interquantile_range
  return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in df.columns:
    outliers_col = check_outlier(df, col)
    print(f"{col} sütunu aykırı değerler var : {outliers_col}")

def replace_with_thresholds(dataframe, col):
    low_limit, up_limit = outlier_thresholds(dataframe, col)
    dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
    dataframe.loc[(dataframe[col] > up_limit), col] = up_limit

for col in num_cols:
  replace_with_thresholds(df,col)

for col in df.columns:
    outliers_col = check_outlier(df, col)
    print(f"{col} sütunu aykırı değerler var : {outliers_col}")