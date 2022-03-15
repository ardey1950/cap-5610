import pandas as pd #CSV file I/O, data processing
import numpy as np #mean & standard deviation of scores
from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score #evaluating performance of a classifier, including K-fold Cross Validation
from sklearn import svm #Support Vector Machine
from sklearn.svm import SVC #Support Vector Classification
        
#reading training data from CSV file
train_data = pd.read_csv("./titanic/train.csv")

#dropping records with missing value of label
train_data.dropna(axis=0, subset=['Survived'], inplace=True)
#label
y = train_data["Survived"]

#selected features
features = ["Pclass", "Sex", "SibSp", "Parch"]
#conversion from categorical variable to indicator/dummy variable - "Sex" attribute
X = pd.get_dummies(train_data[features])

#setting up 5-fold cross-validation, with shuffling before splitting
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cross_validate(model):
    scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    print("Accuracy: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

for i in ["linear", "poly", "rbf"]:
    print("SVM Kernel {}:" .format(i))
    cross_validate(svm.SVC(kernel=i))
    print("")
