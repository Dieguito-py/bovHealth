import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

data = pd.read_csv('Data/Raw/dados1.csv')

X = data[['x', 'y', 'z']]
y = data['atividade']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

os.makedirs('data/processed/', exist_ok=True)

pd.DataFrame(X_train, columns=['x', 'y', 'z']).to_csv('Data/Processed/X_train.csv', index=False)
pd.DataFrame(X_test, columns=['x', 'y', 'z']).to_csv('Data/Processed/X_test.csv', index=False)
y_train.to_csv('Data/Processed/y_train.csv', index=False)
y_test.to_csv('Data/Processed/y_test.csv', index=False)
