import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

full_df = pd.read_csv("C:/Users/Eddie/Desktop/full_df_2.csv")
half_df = full_df.loc["2016-01-01 00:00:00":"2023-03-12 00:00:00", :]
half_df = full_df[['현재수요(MW)', '공급능력(MW)', '최대예측수요(MW)', '공급예비력(MW)', '공급예비율(퍼센트)',
       '운영예비력(MW)', '운영예비율(퍼센트)', 'mean_temp', 'wind_1', 'wind_2',
       'wind_3', 'wind_4',
       ]]
#half_df.loc[:, "현재수요(MW)"] = MinMaxScaler().fit_transform(np.array(half_df.loc[:, "현재수요(MW)"]).reshape(-1, 1))
del full_df
#print(half_df.shape)

#Time Series Spliiter

def custom_splitter(X, train_size = 512, test_size = 72, step = 10000, enex = True):
  #X.shape = [1140072, 12]
  X = X.astype("float32") #정보형 바꿔서 용량 낮추기
  #X = X.astype("float16") #정보형 바꿔서 용량 낮추기


  #train_size = 내가 설정 가능. 일단 default 는 512
  #test_size = 72. 문제에서 하라고 한 게 이거니까


  len = X.shape[0]
  maginot = len - (train_size + test_size)
  X_array = np.zeros(((maginot//step)+1, train_size, X.shape[1]))

  if enex:
    y_array = np.zeros(((maginot//step)+1, test_size, X.shape[1]))

    for i, m in enumerate(range(0, maginot, step)): #30은 그냥 랜덤한 값. window 간 떨어져 있는 거리. 리소스 다운 때문.
      start_idx = m
      X_end_idx = start_idx + train_size
      y_end_idx = X_end_idx + test_size

      X_data = np.array(X.iloc[start_idx:X_end_idx, :]) #X_data : (512, #features + 1)
      y_data = np.array(X.iloc[X_end_idx:y_end_idx, :]) #y_data : (72, #features + 1)

      X_array[i] = X_data
      y_array[i] = y_data

  else:
    y_array = np.zeros(((maginot//step)+1, test_size))

    for i, m in enumerate(range(0, maginot, step)): #30은 그냥 랜덤한 값. window 간 떨어져 있는 거리. 리소스 다운 때문.
      start_idx = m
      X_end_idx = start_idx + train_size
      y_end_idx = X_end_idx + test_size

      X_data = np.array(X.iloc[start_idx:X_end_idx, :]) #X_data : (512, #features + 1)
      y_data = np.array(X.iloc[X_end_idx:y_end_idx, 0]) #y_data : (72, #features + 1)

      X_array[i] = X_data
      y_array[i] = y_data

  return X_array, y_array


"""
X, Y = custom_splitter(half_df, train_size = 512, test_size = 72, step = 12)
print(X.shape, Y.shape)
"""


def endog_exog(X_splitted, y_splitted):
  X_endog = X_splitted[:, :, 0]
  X_exog = X_splitted[:, :, 1:]
  y_endog = y_splitted[:, :, 0]
  #y_exog = y_splitted[: , :, 1:]

  return X_endog, X_exog, y_endog



"""
X_endog, X_exog, y_endog = endog_exog(X, Y)
print(X_endog.shape, X_exog.shape, y_endog.shape)

#(12384, 216) (12384, 216, 51) (12384, 72)

"""

def time_train_test(X_endogeneous, X_exogeneous, y_endogeneous, test_prop = 0.2):
  length = X_endogeneous.shape[0]
  train_thr = int(np.round(length*(1-test_prop)))
  perm = np.random.permutation(length)
  train_idx = perm[:train_thr]
  test_idx = perm[train_thr:]


  X_endog_train = X_endogeneous[train_idx]
  X_exog_train = X_exogeneous[train_idx]

  y_endog_train = y_endogeneous[train_idx]

  X_endog_test = X_endogeneous[test_idx]
  X_exog_test = X_exogeneous[test_idx]

  y_endog_test = y_endogeneous[test_idx]


  return X_endog_train, X_exog_train, y_endog_train, X_endog_test, X_exog_test, y_endog_test

def dataset_pipeline(df, train_size, test_size, step = 13, test_prop = 0.2, split = False, enex = True):
  X, Y = custom_splitter(df, train_size, test_size, step, enex)
  X_endog, X_exog, y_endog = endog_exog(X, Y)

  if split:

    X_endog_train, X_exog_train, y_endog_train, X_endog_test, X_exog_test, y_endog_test = time_train_test(X_endog, X_exog, y_endog, test_prop)


    return X_endog_train, X_exog_train, y_endog_train, X_endog_test, X_exog_test, y_endog_test

  else:

    return X_endog, X_exog, y_endog
