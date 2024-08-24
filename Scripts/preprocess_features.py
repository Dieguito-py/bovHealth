import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

data = pd.read_csv('Data/Features/features_treinamento2.csv')

X = data[['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', 'range_x', 'range_y', 'range_z', 'mean_magnitude', 'energy_x', 'energy_y', 'energy_z']]
y = data['atividade']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

os.makedirs('data/processed/', exist_ok=True)

pd.DataFrame(X_train, columns=['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', 'range_x', 'range_y', 'range_z', 'mean_magnitude', 'energy_x', 'energy_y', 'energy_z']).to_csv('Data/Processed/X_train.csv', index=False)
pd.DataFrame(X_test, columns=['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', 'range_x', 'range_y', 'range_z', 'mean_magnitude', 'energy_x', 'energy_y', 'energy_z']).to_csv('Data/Processed/X_test.csv', index=False)
y_train.to_csv('Data/Processed/y_train.csv', index=False)
y_test.to_csv('Data/Processed/y_test.csv', index=False)
