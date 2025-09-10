# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select features and targets, and split into training and testing sets.
2. Scale both X (features) and Y (targets) using StandardScaler.
3. Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.
4. Predict on test data, inverse transform the results, and calculate the mean squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SUSITHRA.B
RegisterNumber:  212223220113

*/
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a sample dataset
data = {
    'area': [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'rooms': [2, 3, 3, 4, 4, 5, 5, 6],
    'age': [5, 10, 15, 20, 8, 12, 18, 25],
    'location_score': [8, 7, 6, 9, 7, 8, 9, 6],
    'price': [200000, 250000, 300000, 400000, 450000, 500000, 550000, 600000],
    'occupants': [3, 4, 4, 5, 6, 6, 7, 8]
}

df = pd.DataFrame(data)

# Step 2: Features and targets
X = df[['area', 'rooms', 'age', 'location_score']]
y_price = df['price']
y_occupants = df['occupants']

# Step 3: Split dataset
X_train, X_test, y_price_train, y_price_test, y_occ_train, y_occ_test = train_test_split(
    X, y_price, y_occupants, test_size=0.3, random_state=42
)

# Step 4: Train SGDRegressor for price prediction
price_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
price_model.fit(X_train, y_price_train)

# Step 5: Train SGDRegressor for occupants prediction
occ_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
occ_model.fit(X_train, y_occ_train)

# Step 6: Predictions
y_price_pred = price_model.predict(X_test)
y_occ_pred = occ_model.predict(X_test)

# Step 7: Evaluation
print("Price Prediction:")
print("MSE:", mean_squared_error(y_price_test, y_price_pred))
print("R2 Score:", r2_score(y_price_test, y_price_pred))

print("\nOccupants Prediction:")
print("MSE:", mean_squared_error(y_occ_test, y_occ_pred))
print("R2 Score:", r2_score(y_occ_test, y_occ_pred))

# Show predictions vs actual
results = pd.DataFrame({
    'Actual Price': y_price_test,
    'Predicted Price': y_price_pred,
    'Actual Occupants': y_occ_test,
    'Predicted Occupants': y_occ_pred
})

print("\nPrediction Results:")
print(results)

```

## Output:
<img width="650" height="243" alt="image" src="https://github.com/user-attachments/assets/6c8fd98d-ca50-4708-8e7c-757c87fb282b" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
