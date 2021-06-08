# Import Library
import numpy as np
import pandas as pd

# Data Load
veriler = pd.read_excel("Odev_tenis.xlsx")
print("\n************** Odev_tenis Dataset **************\n")
print(veriler)

# Import Library
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Encoder: [Categorical -> Numeric] All Column

# Label Encoding Map on Data
veriler2 = veriler.apply(LabelEncoder().fit_transform)
print("\n\n********** Label Encoding Map on Data (veriler2) **********\n")
print(veriler2)

# Outlook Datasets Label Encode to Change One Hot Encode

# Label Encode
outlookDataset = veriler2.iloc[:, :1]

# One Hot Encode
ohe = OneHotEncoder()
outlookDataset = ohe.fit_transform(outlookDataset).toarray()

print("\n\n********** Outlook Dataset Label Encode to Change One Hot Encode **********\n")
print(outlookDataset)

# Combine Datasets ( veriler , veriler2 => sonVeriler)
havaDurumu = pd.DataFrame(data=outlookDataset, index=range(14), columns=["overcast", "rainy", "sunny"])
sonVeriler = pd.concat([havaDurumu,veriler.iloc[:,1:3]],axis=1)
sonVeriler = pd.concat([veriler2.iloc[:,-2:],sonVeriler],axis=1)
print("\n\n********** Combine Datasets (veriler,veriler2) and New Table (sonVeriler) **********\n")
print(sonVeriler)

#Training Dataset

# Import Library
from sklearn.model_selection import train_test_split

# Training Humidity Data and Testing 'sonVeriler' Dataset
x_train,x_test,y_train,y_test=train_test_split(sonVeriler.iloc[:,:-1],sonVeriler.iloc[:,-1:],test_size=0.33,random_state=0)
print("\n\n********** x_train **********\n")
print(x_train)
print("\n\n********** x_test **********\n")
print(x_test)
print("\n\n********** y_train **********\n")
print(y_train)
print("\n\n********** y_test **********\n")
print(y_test)

# Import Library
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Train Regressor with x_train and y_train
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

# Predict Humidity Data
print("\n\n********** Predict Humidity Data **********\n")
print(y_pred)


# Backward Elimination

# Import Library
import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=sonVeriler.iloc[:,:-1],axis=1)
print("\n\n********** SonVeriler Numpy Array Type **********\n")
print(X)

# [windy,play,overcast,rainy,sunny,temperature] Column
X_l = sonVeriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)

# Humidity Column , [windy,play,overcast,rainy,sunny,temperature] Column
r_ols = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()
print(r_ols.summary())
#print(X_l)

# Remove Windy Column on 'sonVeriler' Dataset
# Note : You can remove the x with a high P-Value and look at the system's recovery again.
sonVeriler = sonVeriler.iloc[:,1:]

X_l = sonVeriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
r_ols = sm.OLS(sonVeriler.iloc[:,-1:],X_l).fit()
print(r_ols.summary())
#print(X_l)

# As a result of the Backward Elimination method , we were able to predict results that approached the real values.
# I removed the x with a high P-Value and look at the system's recovery again.

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

# Train 'Regressor' with new x_train and new y_train
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print("\n\n********** New Predict Humidity Data **********\n")
print(y_pred)

