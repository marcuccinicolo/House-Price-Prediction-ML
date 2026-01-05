import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score

# 1. Load the data and the trained model
df = pd.read_csv("/Users/nicomarcucci/Desktop/HOUSE PROJECT GH/data/cleaned_housing_data.csv")
model = joblib.load('house_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Prepare the test data (same split as before)
from sklearn.model_selection import train_test_split
X = df.drop('price', axis=1)
y = df['price']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the test data and make predictions
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

# 4. PLOT: Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Linea di perfezione
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted House Prices (R2 = {r2_score(y_test, predictions):.3f})')
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# 5. FEATURE IMPORTANCE: Quali variabili contano di più?
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Top 10 Most Important Features for House Price')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Charts saved in OUTPUT_CHARTS folder!")