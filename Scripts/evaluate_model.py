# Gerar previsões e avaliar o modelo ajustado

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

X_test = pd.read_csv('Data/Processed/X_test.csv')
y_test = pd.read_csv('Data/Processed/y_test.csv').values.ravel()

with open('Models/model_vaca_01.pkl', 'rb') as file:
    model = pickle.load(file)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo ajustado: {accuracy:.2f}')
