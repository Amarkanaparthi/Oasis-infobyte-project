import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')
print(data.head())

print(data.tail())

print(data.isnull().sum())

data.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate",
               "Estimated Employed",
               "Estimated Labour Participation Rate",
               "Region","longitude","latitude"]


numeric_data = data.select_dtypes(include=[float, int])

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


data.columns = ["States", "Date", "Frequency",
                "Estimated Unemployment Rate", "Estimated Employed",
                "Estimated Labour Participation Rate", "Region",
                "longitude", "latitude"]

# Create a new figure
plt.figure(figsize=(12, 8))

# Set the title
plt.title("Indian Unemployment")

# Create a histogram plot with different regions colored
sns.histplot(x="Estimated Employed", hue="Region", data=data)

# Display the figure
plt.show()