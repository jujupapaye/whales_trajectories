import soundfile as sf  
import pandas as pd
from scipy import signal
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# corresponding hydrophones and channels on data
hydrophones = {'A': 2, 'B': 1, 'C':0}

def get_pulse(path, t, channel):
    data_folder = '/nfs/NAS6/SABIOD/SITE/KM3Net/DATA_WAV/'
    x, fs = sf.read(data_folder + path)
    x = x[int(t*fs-fs//2):int(t*fs+fs//2),channel]
    return x, fs

# resample at 4kHz
def resample(sig, fs):
    resampled_sig = signal.resample(sig, int(len(sig)/fs*4000)) 
    return resampled_sig

# bandpass filter between 15 and 25 Hz to keep just the pulse of fin whales
def filter(sig,fs):
    sos = signal.butter(2, [15,25], 'bandpass', fs=fs, output='sos')  # create filter
    filtered = signal.sosfilt(sos, sig)  # filter the sound
    return filtered

# open and get data without outliers
def open_data():
    data_folder = 'data/'
    data = pd.read_pickle(data_folder + 'locations_xcorr_regression.pkl')
    data = data[(data['x'] >= -20000) & (data['x'] <= 20000) & (data['y'] >= -20000) & (data['y'] <= 33000)] # filter outliers
    return data

# get sound of just one hydrophone, hydro has to be 'A', 'B' or 'C'
def get_data_array(hydro='A'):
    data = open_data()
    channel = hydrophones[hydro]
    all_data = np.zeros((len(data), 195312))
    for sound_filename, i, t_i in zip(data['fn'], np.arange(len(data)), data['time']):
        print(i)
        folder = sound_filename[:7]
        sig, fs = get_pulse(folder + '/' + sound_filename, t_i, channel)
        sig = filter(sig, fs)
        #sig = resample(sig, fs)
        all_data[i] = sig
    return all_data

# get just the positions of the fin whales
def get_positions():
    data = open_data()
    all_positions = np.zeros((len(data), 2))  # x,y
    for x,y,i in zip(data['x'],data['y'],np.arange(len(data))):
        all_positions[i,0] = x
        all_positions[i,1] = y
    return all_positions

def get_times():
    data = open_data()
    all_times = np.zeros(len(data))
    all_dates = []
    for t,d,i in zip(data['time'], data['date'] ,np.arange(len(data))):
        all_times[i] = t
        all_dates.append(d)
    return all_times, all_dates



