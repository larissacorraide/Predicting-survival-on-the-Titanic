# Libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn import preprocessing

# Reading the train data file
df = pd.read_csv('train.csv')

#Pclass, Sex, Age to predict whether or not a passenger survived.
df_useful = df[['Pclass', 'Sex', 'Age','Survived']] #Selecting the needed columns

df_useful.loc[df_useful['Sex'] == 'male', 'Sex'] = 0 # changing male =0 and famale =1
df_useful.loc[df_useful['Sex'] == 'female', 'Sex'] = 1

mean = df_useful['Age'].mean()
df_useful['Age'] = df_useful['Age'].fillna(mean)

x = df_useful.drop('Survived', axis = 1) # Input = Age, Sex, Pclass
y= df_useful['Survived']                # Output = Surived or not

x = preprocessing.normalize(x)

# Test
df_test = pd.read_csv('test.csv')
#df_test = df_test.dropna() # remove null values

#Pclass, Sex, Age to predict whether or not a passenger survived.
df_useful_test = df_test[['Pclass', 'Sex', 'Age']] #Selecting the needed columns

#mean = df_useful_test['Age'].mean()
df_useful_test['Age'] = df_useful_test['Age'].fillna(mean)

mean_Class = df_useful_test['Pclass'].mean()
df_useful_test['Pclass'] = df_useful_test['Pclass'].fillna(mean_Class)

df_useful_test_index = df_test[['PassengerId']]
df_useful_test.loc[df_useful_test['Sex'] == 'male', 'Sex'] = 0 # changing male =0 and famale =1
df_useful_test.loc[df_useful_test['Sex'] == 'female', 'Sex'] = 1

df_useful_test = preprocessing.normalize(df_useful_test)
# Input = Age, Sex, Pclass


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.35, random_state= 0)

# Parameter that we want to tune for each method.
tuned_parameters_tree = [{'criterion': ['gini','entropy'], 'max_depth': [1,2,3,4,5,6,7,8,9,10]}]
tuned_parameters_forest = [{'n_estimators': [1,2,10,50,100,200], 'max_depth': [1,2,5,10,15],'criterion':['gini','entropy']}]
tuned_parameters_SVM = [{'C': [0.001,0.1,1.0,10,100,500], 'kernel': ['linear', 'poly', 'rbf','sigmoid'],'degree':[1,2,3]}]
scores = ['f1']


# Trying to get the best parameters based on the precision score of the model.
for score in scores:

    model_tree = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters_tree, cv=3, scoring='%s' % score, verbose=10)
    model_tree.fit(x_train, y_train) # Creating the decision tree model

    model_forest = GridSearchCV(RandomForestClassifier(), tuned_parameters_forest, cv=3, scoring='%s' % score, verbose=10)
    model_forest.fit(x_train, y_train) # Creating the random forest model

    model_SVM = GridSearchCV(SVC(), tuned_parameters_SVM, cv=3, scoring='%s' % score, verbose=10)
    model_SVM.fit(x_train, y_train) # Creating SVM model


file = open("parameters.txt", "w+")
file.write("Best parameters set found on development set (tree):")
file.write("\n")
file.write(str(model_tree.best_params_))
file.write("\n")
file.write("\n")
file.write("Best parameters set found on development set (forest):")
file.write("\n")
file.write(str(model_forest.best_params_))
file.write("\n")
file.write("\n")
file.write("Best parameters set found on development set (SVM):")
file.write("\n")
file.write(str(model_SVM.best_params_))
file.close()


y_true_tree, y_pred_tree = y_test, model_tree.predict(x_test) # Test the model created by using the same data
y_true_forest, y_pred_forest = y_test, model_forest.predict(x_test)  # Test the model created by using the same data
y_true_SVM, y_pred_SVM = y_test, model_SVM.predict(x_test)  #Test the model created by using the same data

# Printing the classification report for each model
print(classification_report(y_true_tree, y_pred_tree))
print()
print(classification_report(y_true_forest, y_pred_forest))
print()
print(classification_report(y_true_SVM, y_pred_SVM))

# Printing the accuracy score for each model
print(round (accuracy_score(y_test, y_pred_tree),2))
print(round (accuracy_score(y_test, y_pred_forest),2))
print(round (accuracy_score(y_test, y_pred_SVM),2))
print()

# Calculating ao the F1_scores for each model
F1_score_arvore = round(f1_score(y_test, y_pred_tree),2)
F1_score_forest = round (f1_score(y_test, y_pred_forest),2)
F1_score_SVM = round (f1_score(y_test, y_pred_SVM),2)


#Printing the F1_score for each model
print(F1_score_arvore)  # Acuracia do modelo forest
print(F1_score_forest)  # Acuracia do modelo SVM
print(F1_score_SVM)

#Creating a data frame dor the F1_score for all methods
data = [F1_score_arvore,F1_score_forest,F1_score_SVM]
F1_score_all = pd.DataFrame(data, index=['Tree','Forest','SVM'], columns=['F1_score'])


# Ploting a chart F1_score vs Method
F1_score_all["F1_score"].plot(kind='bar', color = ['blue'],  width=0.3)
plt.title("F1_score vs Method")
plt.xlabel("Method")
plt.ylabel("F1_score")
plt.xticks(rotation='vertical')
plt.yticks(np.arange(0.5, 1.1, step=0.05))
plt.ylim(0.5,1.0)
plt.show()


#Prediction

y_pred_tree_final = model_tree.predict(df_useful_test) # Test the model created by using the same data
y_pred_forest_final = model_forest.predict(df_useful_test)  # Test the model created by using the same data
y_pred_SVM_final = model_SVM.predict(df_useful_test)  #Test the model created by using the same data

submission_tree = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred_tree_final})
submission_tree.set_index('PassengerId', inplace = True)
print(submission_tree)
submission_tree.to_csv("prediction_tree.csv",encoding="utf-8")

submission_forest = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred_forest_final})
submission_forest.set_index('PassengerId', inplace = True)
submission_forest.to_csv("prediction_forest.csv",encoding="utf-8")

submission_SVM = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred_SVM_final})
submission_SVM.set_index('PassengerId',inplace = True)
submission_SVM.to_csv("prediction_SVM.csv",encoding="utf-8")
