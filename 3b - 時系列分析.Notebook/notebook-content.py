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
# META     },
# META     "environment": {
# META       "environmentId": "8f2ed9fd-4adb-4019-83c4-69957fa491c1",
# META       "workspaceId": "b0415ddc-8fba-41ba-aa70-b251628a9f82"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 不動産取引価格の時系列分析と予測

# CELL ********************

LAKEHOUSE_TABLE_PATH = "abfss://<workspace_id>@onelake.dfs.fabric.microsoft.com/<lakehouse_id>/Tables/"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# マジックコマンドによるライブラリのインストール
%pip install statsmodels

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# MLflow の設定
import mlflow
mlflow.set_experiment("realestate-transaction-forecast")

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

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 東京都の不動産価格にフォーカス
tokyo = df.loc[df['Prefecture'] == '東京都']
print(tokyo['Date'].min(), tokyo['Date'].max())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# データの前処理
cols = ['Id', 'Year', 'Quarter', 'PriceCategory', 'Type', 'Region', 'MunicipalityCode', 'Prefecture', 'Municipality', 'DistrictName', 'TradePrice', 'Area', 'UnitPrice', 'LandShape', 'Purpose', 'Frontage', 'Direction', 'Classification', 'Breadth', 'CityPlanning', 'CoverageRatio', 'FloorAreaRatio', 'Remarks']

tokyo.drop(cols, axis=1, inplace=True)
tokyo['Date'] = pd.to_datetime(tokyo['Date'])
tokyo = tokyo.sort_values('Date')
tokyo.isnull().sum()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# データの準備
tokyo = tokyo.groupby('Date')['PricePerUnit'].mean().reset_index()
tokyo = tokyo.set_index('Date')
tokyo.index
y = tokyo
maximim_date = y.reset_index()['Date'].max()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 坪単価の推移をプロット
from matplotlib.ticker import ScalarFormatter
ax = y.plot(figsize=(12, 3))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# ハイパーパラメーター チューニング
import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit(disp=False)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# モデルのトレーニング
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit(disp=False)
print(results.summary().tables[1])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 実績データに対する予測
predictions = results.get_prediction(start=maximim_date-pd.DateOffset(years=5), dynamic=False)
# 将来のデータに対する予測
predictions_future = results.get_prediction(start=maximim_date+ pd.DateOffset(months=3),end=maximim_date+ pd.DateOffset(years=5),dynamic=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# レポートによる可視化のためのデータ準備
Future = pd.DataFrame(predictions_future.predicted_mean).reset_index()
Future.columns = ['Date','Forecasted_Sales']
Future['Actual_Sales'] = np.NAN
Actual = pd.DataFrame(predictions.predicted_mean).reset_index()
Actual.columns = ['Date','Forecasted_Sales']
y_truth = y['2018-10-01':]
Actual['Actual_Sales'] = y_truth.values
final_data = pd.concat([Actual,Future])
# Calculate the mean absolute percentage error (MAPE) between 'Actual_Sales' and 'Forecasted_Sales' 
final_data['MAPE'] = mean_absolute_percentage_error(Actual['Actual_Sales'], Actual['Forecasted_Sales']) * 100
final_data['Prefecture'] = "東京都"
final_data[final_data['Actual_Sales'].isnull()]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

input_df = y.reset_index()
input_df.rename(columns = {'Date':'Date','PricePerUnit':'Actual_Sales'}, inplace=True)
input_df['Prefecture'] = '東京都'
input_df['MAPE'] = np.NAN
input_df['Forecasted_Sales'] = np.NAN

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# レポート用のテーブルへデータを格納

final_data_2 = pd.concat([input_df,final_data])
final_data_2['Date'] = final_data_2['Date'].dt.date

table_name = "RealEstateForecast"
write_deltalake(LAKEHOUSE_TABLE_PATH + table_name, final_data_2, mode="overwrite")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
