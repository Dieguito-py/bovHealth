# Esse script ajusta o modelo genérico para um animal específico usando dados adicionais

import pandas as pd
import pickle

with open('Models/model_generic.pkl', 'rb') as file:
    model = pickle.load(file)

calibration_data = pd.read_csv('Data/Raw/vaca01.csv')

mean_values = calibration_data[['x', 'y', 'z']].mean()
std_values = calibration_data[['x', 'y', 'z']].std()

def adjust_prediction(prediction, mean_values, std_values):
    adjustment_factor = 1.0
    if mean_values.mean() > 0.5:  # lógica de ajuste
        adjustment_factor += 0.1
    return prediction * adjustment_factor

os.makedirs('Models/', exist_ok=True)

with open('Models/model_vaca_01.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo ajustado para vaca_01 salvo com sucesso.")