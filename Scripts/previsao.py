
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('models/model_generic.pkl', 'rb') as file:
    model = pickle.load(file)

new_data = pd.read_csv('data/raw/dados3.csv')

X_new = new_data[['x', 'y', 'z']]

scaler = StandardScaler()

X_new_scaled = scaler.fit_transform(X_new)

X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=['x', 'y', 'z'])

predictions = model.predict(X_new_scaled_df)

new_data['predicted_activity'] = predictions
new_data[['x', 'y', 'z', 'predicted_activity', 'predicted_activity']].to_csv('data/processed/novo_dado_com_predicoes.csv', index=False)

print(new_data.head())
