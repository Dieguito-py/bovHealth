import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Data/Raw/dados1.csv')

def butterworth_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Frequência Nyquist
    normal_cutoff = cutoff / nyq  # Normaliza frequência de corte
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

cutoff = 5  # Defina frequência de corte
fs = 50  # Defina taxa de amostragem

df['x'] = butterworth_filter(df['x'], cutoff, fs)
df['y'] = butterworth_filter(df['y'], cutoff, fs)
df['z'] = butterworth_filter(df['z'], cutoff, fs)
df['x'] = df['x'].round(2)
df['y'] = df['y'].round(2)
df['z'] = df['z'].round(2)

df.to_csv('Data/Processed/filtered_data.csv', columns=['x', 'y', 'z', 'atividade'], index=False)
