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

# Decision Tree
from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regressor Class assigned to variable
r_dt = DecisionTreeRegressor(random_state=0)

# Decision Tree Training with 'X' and 'Y'
r_dt.fit(X,Y)

# Visualize the Decision Tree Model
plt.scatter(X,Y,color = "red")
plt.plot(x,r_dt.predict(X),color = "blue")
plt.title("Decision Tree Prediction")
plt.show()

# Prediction by education levels
print("Decision Tree Prediction Abilities\n")
print("11.   Education Levels Salary = ",r_dt.predict([[11]]))
print("6.6.  Education Levels Salary = ",r_dt.predict([[6.6]]),"\n")

# R2 Score
from sklearn.metrics import r2_score
print("Decision Tree R2 Score")
print(r2_score(Y,r_dt.predict(X)))