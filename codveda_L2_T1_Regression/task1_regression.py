import pandas as pd

df = pd.read_csv(
    "4) house Prediction Data Set (1).csv",
    sep=r"\s+",
    header=None
)

df.columns = [
    'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
    'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'
]

print(df.head())

X = df.drop('MEDV', axis=1)
y = df['MEDV']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)






from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully\n")

y_pred = model.predict(X_test)

print("\nPredicted Values:\n", y_pred[:5])

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nModel Coefficients:\n", coef_df)