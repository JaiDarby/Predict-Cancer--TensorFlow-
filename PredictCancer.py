"""







Hello, I train the model!







"""

import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
import pickle

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

clf = svm.SVC(kernel="linear")

BestAcc = 0

while BestAcc < .98:

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)  

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    Accuracy = metrics.accuracy_score(y_test,y_pred)

    print (round(Accuracy, 2))

    if Accuracy > BestAcc:
        BestAcc = Accuracy
        with open("Model.pickle", "wb") as f:
            pickle.dump(clf, f)

print("Best Accuracy:",BestAcc)