import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("Salary_dataset.csv")


x=np.array(df['YearsExperience'])
x=x.reshape(30,1)

y=np.array(df['Salary'])
y=y.reshape(30,1)

plt.scatter(x,y)

#Prediction
x_mean=np.mean(x)
y_mean=np.mean(y)

#slope calculation

m=(((x-x_mean)*(y-y_mean)).sum())/(((x-x_mean)**2).sum())

#intercept calculation

c=(y_mean-(m*x_mean))


def predict(value):
    return (value*m)+c

x_df=pd.DataFrame(x)
y_predicted=x_df.apply(predict)
y_predicted=np.array(y_predicted)

plt.plot(x,y_predicted)
plt.show()
