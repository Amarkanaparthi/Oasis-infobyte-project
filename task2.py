import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("CarPrice.csv")
print(data)

print(data.isnull().sum())

print(data.describe())

data.Car_Name.unique()

sns.set_style("whitegrid")

# Create Figure 1
plt.figure(figsize=(15, 10))

# Create a distribution plot for the "price" column in your data
sns.distplot(data.Present_Price)

# Set the title for Figure 1
plt.title("Figure 1: Distribution of Price")

# Display Figure 1
plt.show()


plt.figure(figsize=(20, 15))

# Calculate the correlation matrix
correlations = data.corr()

# Create a heatmap of the correlations
sns.heatmap(correlations, cmap="coolwarm", annot=True)

# Set the title for Figure 2
plt.title("Figure 2: Correlation Heatmap")

# Display Figure 2
plt.show()

predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)