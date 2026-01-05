import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
# Assicurati che il file sia nella cartella DATA/
df = pd.read_csv("/Users/nicomarcucci/Desktop/HOUSE PROJECT GH/data/kc_house_data.csv")

# 2. Basic Cleaning
# Eliminiamo colonne inutili per il modello (ID e data di vendita)
df = df.drop(['id', 'date'], axis=1)
corr_matrix = df.select_dtypes(include=[np.number]).corr()
top_corr_features = corr_matrix.index[abs(corr_matrix["price"]) > 0.2]

# 3. Correlation Analysis
plt.figure(figsize=(15, 8))
sns.heatmap(df[top_corr_features].corr(), 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            linewidths=0.5, 
            annot_kws={"size": 10})
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# 4. Feature Engineering (Esempio: Età della casa)
df['house_age'] = 2024 - df['yr_built']
# Se è stata ristrutturata, usiamo l'anno di ristrutturazione
df['years_since_renovation'] = np.where(df['yr_renovated'] == 0, df['house_age'], 2024 - df['yr_renovated'])

# 5. Visualizing Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=50)
plt.title("Distribution of House Prices")
plt.savefig("price_distribution.png")
plt.show()

# Salva il dataset pulito per il prossimo script
df.to_csv("/Users/nicomarcucci/Desktop/HOUSE PROJECT GH/data/cleaned_housing_data.csv", index=False)
print("Cleaning complete. Data saved to data/cleaned_housing_data.csv")

