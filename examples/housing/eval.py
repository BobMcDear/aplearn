import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('housing.csv')
train, val = train_test_split(df, test_size=0.2, shuffle=True)
X_t, y_t = train.drop(columns=['MedHouseVal']), train['MedHouseVal']
X_v, y_v = val.drop(columns=['MedHouseVal']), val['MedHouseVal']

scaler = StandardScaler()
X_t = scaler.fit_transform(X_t)
X_v = scaler.transform(X_v)

print(mean_squared_error(y_v, Ridge().fit(X_t, y_t).predict(X_v)) ** 0.5)
