'''
Problem Statement
Nowadays, social media advertising is one of the popular forms of advertising. Advertisers can utilise user&#39;s demographic
information and target their ads accordingly.
Implement kNN Classifier to determine whether a user will purchase a particular product displayed on a social network ad
or not.
'''
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df=pd.read_csv("https://raw.githubusercontent.com/VISHNUAJI2002/Data-set/refs/heads/main/social-network-ads.csv")
print(df.head())
print(df.isnull().sum())

features=df.drop(['Purchased','User ID'], axis=1)
target=df['Purchased']

print("Info:",df.info())

# Use 'get_dummies()' function to convert each categorical column in a DataFrame to numerical.
features=pd.get_dummies(features,columns=['Gender'])

xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.2,random_state=42)

# Print the shape of the train and test sets.
print('X_train shape:', xtrain.shape)
print('X_test shape:', xtest.shape)
print('Y_train shape:', ytrain.shape)
print('Y_test shape:', ytest.shape)

# Train kNN Classifier model
knnsocial=KNeighborsClassifier(n_neighbors=3)
knnsocial.fit(xtrain,ytrain)

ytrain_pred=knnsocial.predict(xtrain)
ytest_pred=knnsocial.predict(xtest)

print('Accuracy score of ytrain_pred:', accuracy_score(ytrain,ytrain_pred))
print('Accuracy score of ytest_pred:', accuracy_score(ytest,ytest_pred))

print('classification report:')
print('Train report:',classification_report(ytrain,ytrain_pred))
print('Test report:',classification_report(ytest,ytest_pred))