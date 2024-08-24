import pandas as pd
import numpy as np

df = pd.read_csv('Data/Raw/dados2.csv')

df['index'] = df.index

def normalize_data(df):
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].round(2)
    return df

df = normalize_data(df)

window_size_samples = 20  # 60 segundos / 0,5 segundos
stride_samples = 5        # 10 segundos / 0,5 segundos

def extract_features(window):
    features = {}
    
    # Média
    features['mean_x'] = round(window['x'].mean(), 2)
    features['mean_y'] = round(window['y'].mean(), 2)
    features['mean_z'] = round(window['z'].mean(), 2)
    
    # Desvio Padrão
    features['std_x'] = round(window['x'].std(), 2)
    features['std_y'] = round(window['y'].std(), 2)
    features['std_z'] = round(window['z'].std(), 2)
    
    # Valor Máximo
    features['max_x'] = round(window['x'].max(), 2)
    features['max_y'] = round(window['y'].max(), 2)
    features['max_z'] = round(window['z'].max(), 2)
    
    # Valor Mínimo
    features['min_x'] = round(window['x'].min(), 2)
    features['min_y'] = round(window['y'].min(), 2)
    features['min_z'] = round(window['z'].min(), 2)
    
    # Amplitude
    features['range_x'] = round(features['max_x'] - features['min_x'], 2)
    features['range_y'] = round(features['max_y'] - features['min_y'], 2)
    features['range_z'] = round(features['max_z'] - features['min_z'], 2)
    
    # Magnitude
    magnitude = np.sqrt(window['x']**2 + window['y']**2 + window['z']**2)
    features['mean_magnitude'] = round(magnitude.mean(), 2)
    
    # Energia do Sinal
    features['energy_x'] = round((window['x']**2).sum(), 2)
    features['energy_y'] = round((window['y']**2).sum(), 2)
    features['energy_z'] = round((window['z']**2).sum(), 2)
    
    return features

features_list = []

num_samples = len(df)
start_idx = 0

while start_idx + window_size_samples <= num_samples:
    window = df.iloc[start_idx:start_idx + window_size_samples]
    
    if not window.empty:
        features = extract_features(window)
        features['atividade'] = window['atividade'].mode()[0]  # Usa a moda para representar a atividade na janela
        # features['start_index'] = start_idx
        features_list.append(features)
    
    start_idx += stride_samples


features_df = pd.DataFrame(features_list)

features_df.to_csv('Data/Features/features_treinamento3.csv', index=False)

print("Features extraídas e salvas.")
