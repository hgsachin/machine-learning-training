# Polynomial Regression
#Importing libraries

import numpy as np #Mathematical tools
import matplotlib.pyplot as plt # to plot
import pandas as pd # Import datasets & manage them

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #1:2 is to make the X a matrix
y = dataset.iloc[:, 2].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the LR results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (LR graph)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the PR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial graph)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
