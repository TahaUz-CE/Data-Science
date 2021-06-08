# Import Library
import pandas as pd
import matplotlib.pyplot as plt

# Data Load
AllData = pd.read_excel("satislar.xlsx")
newData = pd.DataFrame(AllData)
print("\n\n********** Satislar DataSets **********\n")
print(newData)

month = newData[["Aylar"]]
print("\n\n********** Month **********\n")
print(month)

Sales = newData[["Satislar"]]
print("\n\n********** Sales **********\n")
print(Sales)

# Import Library
from sklearn.model_selection import train_test_split

#Training Dataset
x_train,x_test,y_train,y_test=train_test_split(month, Sales, test_size=0.33, random_state=0)

# All Trained Data and Test Data
# print("******************************")
# print("x_train")
# print(x_train)
# print("**************")
# print("x_test")
# print(x_test)
# print("******************************")
# print("y_train")
# print(y_train)
# print("**************")
# print("y_test")
# print(y_test)
# print("******************************")

'''
# Data Scaler

# Import Library
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

# print("******************************")
# print(X_train)
# print("******************************")
# print(X_test)
# print("******************************")


#Model Structure

# Import Library
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,Y_train)

tahmin = lr.predict(X_test)
print(tahmin)
'''

# Model Structure (Did not use Data Scaler)

# Import Library
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)
#print(tahmin)
x_train = x_train.sort_index()
y_train = y_train.sort_index()


# Visualizing a Data Model

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Sales Chart by Month")
plt.xlabel("Months")
plt.ylabel("Sales")

plt.show()

