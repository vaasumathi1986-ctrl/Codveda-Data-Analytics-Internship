import pandas as pd
import matplotlib.pyplot as plt
#load dataset
df = pd.read_csv(
    "4) house Prediction Data Set (1).csv",
    sep=r"\s+",
    header=None 
)

# Column Names
df.columns = [
    'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
    'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV' 
]

# First 5 Rows
print(df.head())

# Dataset Info
print(df.info())

# Summary Statistics
print(df.describe())

# Correlation Matrix
print(df.corr())

# Histogram
df['MEDV'].hist()
plt.title("House Price Distribution")
plt.show()

# Box Plot
df.boxplot(column='MEDV')
plt.title("House Price Boxplot")
plt.show()

# Scatter Plot
plt.scatter(df['RM'], df['MEDV'])
plt.xlabel("Number of Rooms")
plt.ylabel("House Price")
plt.title("Rooms vs House Price")
plt.show()


