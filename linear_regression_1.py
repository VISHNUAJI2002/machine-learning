import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/advertising.csv")
print(df.head())

'''
sns.regplot(x="TV",y="Sales",data=df,line_kws={"color":"green"})
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()
'''

x_train,x_test,y_train,y_test=train_test_split(df[["TV"]],df["Sales"],test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train, y_train)

print("Y Intercept (c)=",model.intercept_)
print("Slope=",model.coef_[0])
y_pred=model.predict(x_test)
print("Mean square error:",mean_squared_error(y_test,y_pred))
print("R2 score:",r2_score(y_test,y_pred))

def salesPrediction(n):
    return model.predict([[n]])

n=int(input())
print(salesPrediction(n))

