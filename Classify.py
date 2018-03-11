from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#classifier
classifier_tree = tree.DecisionTreeClassifier();
classifier_svc= SVC();
classifier_GaussianNB = GaussianNB();
classifier_GaussianProcessClassifier = GaussianProcessClassifier();

#training
classifier_tree = classifier_tree.fit(X,Y)
classifier_svc= classifier_svc.fit(X,Y)
classifier_GaussianNB =classifier_GaussianNB.fit(X,Y)
classifier_GaussianProcessClassifier = classifier_GaussianProcessClassifier.fit(X,Y)

test_data = ['male'];
test = [[182, 98, 44]];

prediction_tree = classifier_tree.predict(test);
accuracy_tree = accuracy_score(test_data, prediction_tree) * 100

prediction_svc = classifier_svc.predict(test);
accuracy_svc = accuracy_score(test_data, prediction_svc) * 100

prediction_GaussianNB = classifier_GaussianNB.predict(test);
accuracy_GaussianNB = accuracy_score(test_data, prediction_GaussianNB) * 100

prediction_GaussianProcessClassifier = classifier_GaussianProcessClassifier.predict(test);
accuracy_GaussianProcessClassifier = accuracy_score(test_data, prediction_GaussianProcessClassifier) * 100

#max_accuracy = np.argmax([accuracy_tree, accuracy_svc, accuracy_GaussianNB, accuracy_GaussianProcessClassifier])
print("Prediction for {} is {} with an accuracy of {} with TREE".format(test, prediction_tree, accuracy_tree));
print("Prediction for {} is {} with an accuracy of {} with SVC".format(test, prediction_svc, accuracy_svc));
print("Prediction for {} is {} with an accuracy of {} with GaussianNB".format(test, prediction_GaussianNB, accuracy_GaussianNB));
print("Prediction for {} is {} with an accuracy of {} with GaussianProcessClassifier".format(test, prediction_GaussianProcessClassifier, accuracy_GaussianProcessClassifier));
#print(accuracy_tree)