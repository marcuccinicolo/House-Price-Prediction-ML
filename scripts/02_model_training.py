import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load the cleaned dataset
df = pd.read_csv("/Users/nicomarcucci/Desktop/HOUSE PROJECT GH/data/cleaned_housing_data.csv")

# 2. Define Features (X) and Target (y)
X = df.drop('price', axis=1)
y = df['price']

# 3. Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Professional step)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training: Random Forest
print("Training the Random Forest model... please wait.")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Quick Evaluation
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Performance:")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

# 7. Save the model and the scaler for future use
joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved!")