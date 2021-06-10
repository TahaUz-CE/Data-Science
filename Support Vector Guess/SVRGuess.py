import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DataSets Load
print("************** Maaslar DataSets **************\n")
veriler = pd.read_excel("Maaslar.xlsx")
newVeriler = pd.DataFrame(veriler)
print(newVeriler,"\n")

# Data Frame Divide
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# Array Type Conversion (Pandas(x) => Numpy(X))
X = x.values
Y = y.values
# print(type(x))
# print(type(X))
# print(x)
# print(y)

# DataSets Scale
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
# "X" Data Transform => Scale Data "x_olcekli"
x_olcekli = sc1.fit_transform(X)

# StandardScaler Class
sc2 = StandardScaler()

# "Y" Data Transform => Scale Data "y_olcekli"
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# Support Vector Regression (SVR)
# Note : If you work on Support Vector Regression , you must be scale your data.
from sklearn.svm import SVR

# SVR Class assigned to variable
# If you want to change kernel type => kernel ='linear' or kernel='poly' ...
svr_reg = SVR(kernel="rbf")  # Gaussian Radial Basis Function (RBF)


# SVR Training with 'x_olcekli' and 'y_olcekli'
svr_reg.fit(x_olcekli,y_olcekli)

# Visualize the Support Vector Regression (SVR) Model
plt.scatter(x_olcekli,y_olcekli,color ="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color ="blue")
plt.title("Support Vector Regression Prediction")
plt.show()

# Prediction by education levels
print("Support Vector Regression Prediction Abilities\n")
print("11.  Education Levels Salary [Scaled Value] = ",svr_reg.predict([[11]]))
print("6.6. Education Levels Salary [Scaled Value]= ",svr_reg.predict([[6.6]]),"\n")

# R2 Score
from sklearn.metrics import r2_score
print('Support Vector Regression R2 Score')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)),"\n")