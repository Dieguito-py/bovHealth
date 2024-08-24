import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('models/model_generic.pkl', 'rb') as file:
    model = pickle.load(file)


new_data = pd.read_csv('data/features/features_treinamento3.csv')
X_new = new_data[['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', 'range_x', 'range_y', 'range_z', 'mean_magnitude', 'energy_x', 'energy_y', 'energy_z']]

scaler = StandardScaler()

X_new_scaled = scaler.fit_transform(X_new)

predictions = model.predict(X_new_scaled)

new_data['predicted_activity'] = predictions
new_data[['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', 'range_x', 'range_y', 'range_z', 'mean_magnitude', 'energy_x', 'energy_y', 'energy_z', 'predicted_activity']].to_csv('data/processed/novo_dado_com_predicoes.csv', index=False)

print(new_data.head())
