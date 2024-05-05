import pandas as pd
from sklearn.decomposition import PCA

#Electricity Demand Dataset
full_df = pd.read_csv("C:/Users/Eddie/Desktop/full_df.csv")
full_df.drop(["Unnamed: 0"], axis = 1, inplace = True)
full_df.index = pd.to_datetime(full_df["기준일시"])
full_df.drop(["기준일시"], axis = 1, inplace = True)

#Weather Dataset
weather_colname = ['청주 기온(°C)', '청주 강수량(mm)', '청주 풍속(m/s)', '창원 기온(°C)', '창원 강수량(mm)',
       '창원 풍속(m/s)', '제주 기온(°C)', '제주 강수량(mm)', '제주 풍속(m/s)', '전주 기온(°C)',
       '전주 강수량(mm)', '전주 풍속(m/s)', '인천 기온(°C)', '인천 강수량(mm)', '인천 풍속(m/s)',
       '원주 기온(°C)', '원주 강수량(mm)', '원주 풍속(m/s)', '울산 기온(°C)', '울산 강수량(mm)',
       '울산 풍속(m/s)', '안동 기온(°C)', '안동 강수량(mm)', '안동 풍속(m/s)', '수원 기온(°C)',
       '수원 강수량(mm)', '수원 풍속(m/s)', '서울 기온(°C)', '서울 강수량(mm)', '서울 풍속(m/s)',
       '서귀포 기온(°C)', '서귀포 강수량(mm)', '서귀포 풍속(m/s)', '부산 기온(°C)', '부산 강수량(mm)',
       '부산 풍속(m/s)', '대전 기온(°C)', '대전 강수량(mm)', '대전 풍속(m/s)', '대구 기온(°C)',
       '대구 강수량(mm)', '대구 풍속(m/s)', '광주 기온(°C)', '광주 강수량(mm)', '광주 풍속(m/s)']

wind = [] ; rain = [] ; temp = []

for i in range(len(weather_colname)):
  if i % 3 == 0:
    temp.append(weather_colname[i])
  elif i % 3 == 1:
    rain.append(weather_colname[i])
  elif i % 3 == 2:
    wind.append(weather_colname[i])

#Temperature Information Engineering

temp_df = full_df[temp]
full_df = full_df.drop(temp, axis = 1)
temp_2 = temp_df.mean(axis = 1)
full_df["mean_temp"] = temp_2

del temp_df

#Wind Information Engineering

wind_df = full_df[wind]
pca = PCA(n_components = 4)
components_4 = pca.fit_transform(wind_df)
components_4 = pd.DataFrame(components_4)
components_4.columns = ["wind_1", "wind_2", "wind_3", "wind_4"]
components_4.index = wind_df.index
full_df.drop(wind, axis = 1, inplace = True)
full_df = pd.concat([full_df, components_4], axis = 1)

#Rain Information Engineering

rain_df = full_df[rain]
pca_rain = PCA(n_components = 4)
rain_4 = pca_rain.fit_transform(rain_df)
#print("4 components of rain: ", pca_rain.explained_variance_ratio_)

rain_4 = pd.DataFrame(rain_4)
rain_4.columns = ['rain_1', "rain_2", "rain_3", "rain_4"]
rain_4.index = full_df.index
full_df.drop(rain, axis = 1, inplace = True)
full_df = pd.concat([full_df, rain_4], axis = 1)
full_df_2 = full_df.drop(["rain_1", "rain_2", "rain_3", "rain_4"], axis = 1)

del rain_df, wind_df, full_df

print(full_df_2.head())
full_df_2.to_csv("C:/Users/Eddie/Desktop/full_df_2.csv")