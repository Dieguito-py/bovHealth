import pandas as pd
import numpy as np

df = pd.read_csv('Data/Raw/dados.csv')

# x, y, z
df['index'] = df.index


window_size_samples = 120  # 60 segundos / 0,5 segundos
stride_samples = 20        # 10 segundos / 0,5 segundos

def extract_features(window):
    features = {}
    
    # Média
    features['mean_x'] = window['x'].mean()
    features['mean_y'] = window['y'].mean()
    features['mean_z'] = window['z'].mean()
    
    # Desvio Padrão
    features['std_x'] = window['x'].std()
    features['std_y'] = window['y'].std()
    features['std_z'] = window['z'].std()
    
    # Valor Máximo
    features['max_x'] = window['x'].max()
    features['max_y'] = window['y'].max()
    features['max_z'] = window['z'].max()
    
    # Valor Mínimo
    features['min_x'] = window['x'].min()
    features['min_y'] = window['y'].min()
    features['min_z'] = window['z'].min()
    
    # Amplitude
    features['range_x'] = features['max_x'] - features['min_x']
    features['range_y'] = features['max_y'] - features['min_y']
    features['range_z'] = features['max_z'] - features['min_z']
    
    # Magnitude
    magnitude = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2)
    features['mean_magnitude'] = magnitude.mean()
    
    # Energia do Sinal
    features['energy_x'] = (window['x']**2).sum()
    features['energy_y'] = (window['y']**2).sum()
    features['energy_z'] = (window['z']**2).sum()
    
    return features

features_list = []

num_samples = len(df)
start_idx = 0

while start_idx + window_size_samples <= num_samples:
    window = df.iloc[start_idx:start_idx + window_size_samples]
    
    if not window.empty:
        features = extract_features(window)
        features['start_index'] = start_idx
        features_list.append(features)
    
    start_idx += stride_samples

features_df = pd.DataFrame(features_list)

# Salva o DataFrame em um novo arquivo CSV
features_df.to_csv('Data/Features/features_extracted.csv', index=False)

print("Features extraídas e salvas.")
