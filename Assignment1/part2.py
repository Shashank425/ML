
from statistics import LinearRegression
from matplotlib.pyplot import axis
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LinearRegression
import wget

url = "https://raw.githubusercontent.com/Shashank425/ML/main/mcs_ds_edited_iter_shuffled.csv"
dataFile = wget.download(url)
df = pd.read_csv(dataFile)
#df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")

x = df.iloc[:,0:4]
y = df.iloc[:,4:5]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

LR = LinearRegression()
LR.fit(x_train,y_train)
y_prediction = LR.predict(x_test)
print(y_prediction)
print("R squared=",r2_score(y_test,y_prediction))
print("MSE=",mean_squared_error(y_test,y_prediction))
print("RMSE=",np.sqrt(mean_squared_error(y_test,y_prediction)))