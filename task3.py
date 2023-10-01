import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Advertising.csv")
print(data.head())

print(data.isnull().sum())

plt.figure(figsize=(12, 10))

# Explore the Relationships in Your Data
sns.heatmap(data.corr(), cmap='coolwarm', annot=True, linewidths=0.5)

# Present Your Insights with Style
plt.title("Correlation Heatmap of Your Data")

# Show Your Data's Story
plt.show()

x = np.array(data.drop(["Sales"], axis=1))

# Get the target variable 'y' (Sales)
y = np.array(data["Sales"])

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(xtrain, ytrain)

# Make predictions on the test data
ypred = model.predict(xtest)

# Create a DataFrame to display predicted sales values
data_result = pd.DataFrame(data={"Predicted Sales": ypred})
print(data_result)

