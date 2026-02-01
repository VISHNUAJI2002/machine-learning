'''
Aim: Program to implement decision trees using any standard dataset 
available in the public domain and find the accuracy of the algorithm.
'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/Breast_Cancer.csv")
print(df.head(5))

rows,cols=df.shape
print(f"Rows:{rows},Columns:{cols}")

df.drop(['Unnamed: 32','id'],axis=1,inplace=True)
features=df.drop('diagnosis',axis=1)
target=df['diagnosis']
xtrain,xtest,ytrain,ytest=train_test_split(features,target,train_size=0.8,random_state=42)

model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print("accuracy score:",accuracy_score(ytest,ypred))
print("Classification Report:")
print(classification_report(ytest,ypred))
print("confusion matrix:\n",confusion_matrix(ytest,ypred))

plt.figure(figsize=(10,10))
tree.plot_tree(model,filled=True,fontsize=5)
plt.show()

print(ypred)