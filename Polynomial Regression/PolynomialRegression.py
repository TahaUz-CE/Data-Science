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

# Linear Regression and Model
from sklearn.linear_model import LinearRegression
lin_Reg = LinearRegression()
lin_Reg.fit(X,Y)

# Visualize the Linear Model
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_Reg.predict(X),color = "blue")
plt.title("Linear Regression Prediction")
plt.show()

# Polynomial Regression ve Model

# Second-Order Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
# print(x_poly)
lin_Reg2 = LinearRegression()
lin_Reg2.fit(x_poly,y)

# Visualize the Second-Order Polynomial Model
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_Reg2.predict(poly_reg.fit_transform(X)),color = "blue")
plt.title("Second-Order Polynomial Regression Prediction")
plt.show()

# Fourth-Order Polynomial Regression
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
# print(x_poly3)
lin_Reg3 = LinearRegression()
lin_Reg3.fit(x_poly3,y)

# Visualize the Fourth-Order Polynomial Model
plt.scatter(X,Y,color = "red")
plt.plot(X,lin_Reg3.predict(poly_reg3.fit_transform(X)),color = "blue")
plt.title("Fourth-Order Polynomial Regression Prediction")
plt.show()

# Prediction by education levels
print("\n************** The Regressions Prediction Abilities by Education Levels **************\n")

print("\nLinear Regression\n")
print("11.  Education Levels Salary = ",lin_Reg.predict([[11]]))
print("6.6. Education Levels Salary = ",lin_Reg.predict([[6.6]]),"\n")

print("Second-Order Polynomial Regression\n")
print("11.  Education Levels Salary = ",lin_Reg2.predict(poly_reg.fit_transform([[11]])))
print("6.6. Education Levels Salary = ",lin_Reg2.predict(poly_reg.fit_transform([[6.6]])),"\n")

print("Fourth-Order Polynomial Regression\n")
print("11.  Education Levels Salary = ",lin_Reg3.predict(poly_reg3.fit_transform([[11]])))
print("6.6. Education Levels Salary = ",lin_Reg3.predict(poly_reg3.fit_transform([[6.6]])),"\n")

# Regression Score
from sklearn.metrics import r2_score

print("************** R2 Score **************\n")

print('Linear Regression')
print(r2_score(Y,lin_Reg.predict(X)),"\n")

print('Second-Order Polynomial Regression')
print(r2_score(Y,lin_Reg2.predict(poly_reg.fit_transform(X))),"\n")

print('Fourth-Order Polynomial Regression')
print(r2_score(Y,lin_Reg3.predict(poly_reg3.fit_transform(X))),"\n")