'''
Problem Statement
As an owner of a startup, you wish to forecast the sales of your product to plan how
much money should be spent on advertisements. This is because the sale of a product
is usually proportional to the money spent on advertisements.
Predict the impact of TV advertising on your product sales by performing simple linear
regression analysis.

Activity 1: Analysing the Dataset
Create a Pandas DataFrame for Advertising-Sales dataset using the below link. This
dataset contains information about the money spent on the TV, radio and newspaper
advertisement (in thousand dollars) and their generated sales (in thousand units). The
dataset consists of examples that are divided by 1000.
Dataset Link: https://raw.githubusercontent.com/jiss-sngce/CO_3/main/advertising.csv

Activity 2: Train-Test Split

Activity 3: Model Training
Train the simple regression model using training data to obtain the best fit line
y=mx+c
.
Activity 5: Model Prediction
For the TV advertising of $50,000, what is prediction for Sales? In order to predict this
value, perform the following task:
● Based on the regression line, create a function sales_predicted() which takes
a budget to be used for TV advertising as an input and returns the corresponding
units of Sales.
● Call the function sales_predicted() and pass the amount spent on TV
advertising.

Note: To predict the sales for TV advertising of $50,000, pass 50 as parameter to
sales_predicted() function as the original data of this dataset consists of examples
that are divided by 1000. Also, the value obtained after calling sales_predicted(50)
must be multiplied by 1000 to obtain the predicted units of sales.
'''


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


sns.regplot(x="TV",y="Sales",data=df,line_kws={"color":"green"})
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()


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

