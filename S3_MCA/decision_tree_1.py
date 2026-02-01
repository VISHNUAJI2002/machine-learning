'''
Aim: Program to implement decision trees using any standard dataset 
available in the public domain and find the accuracy of the algorithm.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

df=pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/Iris.csv")
print(df.head(5))

#display the number of rows and columns in dataframe
rows,cols=df.shape
print(f"Rows:{rows}, Columns:{cols}")

featues=df.drop(['Species'],axis=1)
target=df['Species']

xtrain,xtest,ytrain,ytest=train_test_split(featues,target,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(criterion='entropy',min_samples_split=50)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print("accuracy score:",accuracy_score(ytest,ypred))
print("\nclassification report:")
print(classification_report(ytest,ypred))
print("confusion matrix:")
cm=confusion_matrix(ytest,ypred)
print(cm)

plt.figure(figsize=(8,5))
sns.heatmap(cm,annot=True,cmap='Blues')
plt.xlabel("predicted values")
plt.title("Confusion matrix")
plt.show()
