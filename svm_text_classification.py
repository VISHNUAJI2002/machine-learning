'''
Program to implement text classification using Support Vector Machine.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Define the small dataset
# We will classify texts as tech(0) or finance(1)
data = [
    "Apple launched a new iPhone with better neural engine.",  # tech
    "The stock market saw huge gains after the quarterly report.", # finance
    "Google's machine learning model achieved 90% accuracy.",  # tech
    "Investors are worried about rising interest rates and inflation.", # finance
    "Python libraries like scikit-learn are great for ML.", # tech
    "Bonds and treasury yields are highly volatile this week." # finance
]
lables=[0,1,0,1,0,1]

# Split data
xtrain,xtest,ytrain,ytest=train_test_split(data,lables,test_size=0.2,random_state=42)
print(f"Total datapoints:{len(data)}")
print(f"Training datapoints:{len(xtrain)}")
print(f"Testing datapoints:{len(xtest)}")

# Feature Extraction (TF-IDF)
vectorizer=TfidfVectorizer(stop_words='english')

# Convert training data
xtrain_vectors=vectorizer.fit_transform(xtrain)
# Convert testing data
xtest_vectors=vectorizer.transform(xtest)

# Initialize and Train the SVM
model=SVC(kernel='linear',C=1.0,random_state=42)
model.fit(xtrain_vectors,ytrain)

# Predict and Evaluate
ypred=model.predict(xtest_vectors)

print("Accuracy:",accuracy_score(ytest,ypred))
print("Classification Report:\n",classification_report(ytest,ypred))

cm=confusion_matrix(ytest,ypred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.show()

# Simple Prediction
new_text = ["Artificial intelligence is transforming industries."]
new_text_vectorized = vectorizer.transform(new_text)
prediction = model.predict(new_text_vectorized)

if prediction[0] == 0:
    print("\nPrediction: Tech")
else:
    print("\nPrediction: Finance")