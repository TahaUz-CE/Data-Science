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

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor Class assigned to variable
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
# Random Forest Training with 'X' and 'Y'
rf_reg.fit(X,Y.ravel())

# Visualize the Random Forest Model
plt.scatter(X,Y,color = "black")
plt.plot(X,rf_reg.predict(X),color = "red")
plt.title("Random Forest Prediction")
plt.show()

# Prediction by education levels
print("Random Forest Prediction Abilities\n")
print("11.   Education Levels Salary = ",rf_reg.predict([[11]]))
print("6.6.  Education Levels Salary = ",rf_reg.predict([[6.6]]),"\n")

# R2 Score
from sklearn.metrics import r2_score
print("Random Forest R2 degeri ")
print(r2_score(Y,rf_reg.predict(X)))