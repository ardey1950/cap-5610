import pandas as pd #CSV file I/O, data processing
from sklearn import tree #decision tree learning models, including DecisionTreeClassifier
from sklearn import ensemble #ensemble methods for classification, regression & anomaly detection, including RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score #evaluating performance of a classifier, including K-fold Cross Validation
import graphviz #plotting of the decision tree
        
#reading training data from CSV file
train_data = pd.read_csv("./titanic/train.csv")

#printing information summary of training data
print(train_data.info())

#dropping records with missing value of label
train_data.dropna(axis=0, subset=['Survived'], inplace=True)
#label
y = train_data["Survived"]

#selected features
features = ["Pclass", "Sex", "SibSp", "Parch"]
#conversion from categorical variable to indicator/dummy variable - "Sex" attribute
X = pd.get_dummies(train_data[features])

#learn decision tree model
#fine-tuning by pre-pruning - max_depth = 4, min_samples_split is left at its default value of 2
#(max_depth = 4 is chosen to be the same as the number of selected features)
#it is expected the built-in learning algorithm will take care of stopping the splitting process
#when gain (default = Gini) after splitting does not improve
#random_state is set to an integer to obtain deterministic behavior during fitting
clf = tree.DecisionTreeClassifier(max_depth=4, random_state=42)
clf = clf.fit(X, y)

#plot decision tree
dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=list(X.columns),
                                class_names=[str(s) for s in y.unique()],
                                filled=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("decision_tree_model")

#setting up 5-fold cross-validation, with shuffling before splitting
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#the 5-fold cross-validation will be run for both the decision tree model & random forest model,
#iterating over max_depth in the range [1,10], to determine the best accuracy, and
#enabling comparison between decision tree model & random forest model performance
max_depth = [1,2,3,4,5,6,7,8,9,10]

print("\n\n")
print("Computing average classification accuracy with five-fold cross validation of decision tree learning model")
print("---------------------------------------------------------------------------------------------------------")
for val in max_depth:
    score = cross_val_score(tree.DecisionTreeClassifier(max_depth=val, random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores for each fold are: {score} | Average score: {"{:.3f}".format(score.mean())}')
    
print("\n\n")
print("Computing average classification accuracy with five-fold cross validation of random forest learning model")
print("---------------------------------------------------------------------------------------------------------")
for val in max_depth:
    #n_estimators = 100 (default) - the number of trees in the forest
    score = cross_val_score(ensemble.RandomForestClassifier(max_depth=val, random_state= 42), X, y, cv= kf, scoring="accuracy")
    print(f'Scores for each fold are: {score} | Average score: {"{:.3f}".format(score.mean())}')