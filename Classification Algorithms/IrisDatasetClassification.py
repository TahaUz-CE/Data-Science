import pandas as pd

# Images of classification methods of data are available on this site
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# Load Data

veriler = pd.read_excel('Iris.xls')
# print(veriler)

"""

# Other Way for Load Dataset

from sklearn import datasets
iris = datasets.load_iris()

"""

# Independent Variables
x = veriler.iloc[:,1:4].values
# Dependent Variables
y = veriler.iloc[:,4:].values

# print(x)
# print(y)

# Splitting Data for Training and Testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

# Scaling of Data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train.ravel())

y_pred = logr.predict(X_test)

# print("****-x_test-****")
# print(x_test)
# print("****-y_test-****")
# print(y_test)
# print("******Predictions******")
# print(y_pred)

# Confusion Matrix for Logistic Regression

from sklearn.metrics import confusion_matrix

print("Logistic Regression")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# KNN(KNearestNeighbors Classifier)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train.ravel())

y_pred = knn.predict(X_test)

# Confusion Matrix for KNN

print("KNN Predict")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# SVC

from sklearn.svm import SVC

# kernel = rbf,linear,poly ...
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train.ravel())

y_pred = svc.predict(X_test)

# Confusion Matrix for SVC

print("SVC")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train.ravel())

y_pred = gnb.predict(X_test)

# Confusion Matrix for Naive Bayes

print("Naive Bayes")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# 1.Gaussian Naive Bayes
# Use this method if the data or class you are predicting is a continuous value. (real numbers , ...)

# 2.Multinomial Naive Bayes
# This method is used if the data you are estimating is nominal value. (eg. Car brand [Ford,Toyota,Hyundai...])

# 3.Bernoulli Naive Bayes
# If the data contains the case like Binaries then this method is used.(1-0 like 'yes' 'no' binary)


# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train.ravel())

y_pred = dtc.predict(X_test)

# Confusion Matrix for Decision Tree Classifier

print(" Decision Tree Classifier ")
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train.ravel())

y_pred = rfc.predict(X_test)

# Confusion Matrix for Random Forest Classifier

print(" Random Forest Classifier ")
cm = confusion_matrix(y_test,y_pred)
print(cm)