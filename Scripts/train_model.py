import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

X_train = pd.read_csv('Data/Processed/X_train.csv')
y_train = pd.read_csv('Data/Processed/y_train.csv').values.ravel()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs('models/', exist_ok=True)

with open('models/model_generic.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo genérico treinado e salvo com sucesso.")
