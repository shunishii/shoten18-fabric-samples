# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.11"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "e0e959fc-42cd-4286-943d-1db19a3f94c2",
# META       "default_lakehouse_name": "RealEstate",
# META       "default_lakehouse_workspace_id": "ab22d4ad-cff4-4d54-8621-e718af0e745d",
# META       "known_lakehouses": [
# META         {
# META           "id": "e0e959fc-42cd-4286-943d-1db19a3f94c2"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 不動産取引価格の回帰分析

# CELL ********************

LAKEHOUSE_TABLE_PATH = "abfss://<workspace_id>@onelake.dfs.fabric.microsoft.com/<lakehouse_id>/Tables/"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# マジックコマンドによるライブラリのインストール
%pip install deltalake

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# MLflow の設定
import mlflow
mlflow.set_experiment("real-estate-regression")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# Delta Table を Pandas Dataframe に読み込み
import pandas as pd
from deltalake import DeltaTable
dt = DeltaTable(LAKEHOUSE_TABLE_PATH + "Transactions")
df = dt.to_pandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 必要なカラムの選択 (Pandas DataFrame の Data Wrangler によって生成されたコード)
def clean_data(df):
    # 列を選択します: 'Year'、'Type'、およびその他 14 つの列
    df = df.loc[:, ['Year', 'Type', 'Region', \
        'Prefecture', 'PricePerUnit', 'Area', \
        'LandShape', 'Purpose', 'Frontage', \
        'Direction', 'Classification', 'Breadth', \
        'CityPlanning', 'CoverageRatio', 'FloorAreaRatio']]
    return df

df = clean_data(df.copy())
df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 特徴量の作成
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['Type', 'Region', 'Prefecture', 'LandShape', 'Purpose', 'Direction', 'Classification', 'CityPlanning']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

df.head()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 特徴量とターゲット列の分割
X = df.drop(['PricePerUnit'], axis=1)
y = df['PricePerUnit']

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# データセットの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# LightGBM データセットの作成
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# モデルのトレーニング
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data]
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 予測と評価
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 評価
rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
