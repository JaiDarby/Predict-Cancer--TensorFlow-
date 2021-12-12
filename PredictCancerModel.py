"""







Hello, I test the existing model!







"""

import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import pickle

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

#Seperating data in lists to test
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.2)

#Loads a previous model
LoadedPickle = open("Model.pickle", "rb")

clf = pickle.load(LoadedPickle)

#Finding accuracy of test
Accuracy = clf.score(x_test, y_test)

#Finding predictions of test
predictions = clf.predict(x_test)

#Printing the overall accuracy
print ("Accuracy: ", round(Accuracy, 2))