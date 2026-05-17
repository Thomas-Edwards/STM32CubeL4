import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle

def build_prediction(file):
    preds = []
    
    rate, wav = wavfile.read(file)
    
    for i in range(0, wav.shape[0]-config.step, config.step):
        sample = wav[i:i+config.step]
        x = mfcc(sample, rate, numcep=config.nfeat,
                    nfilt=config.nfilt, nfft=config.nfft)
        
        x = (x - np.mean(x)) / np.std(x)

        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        pred = model.predict(x)[0][0]   # sigmoid output
        preds.append(pred)

    # average over all windows
    final_pred = np.mean(preds)

    return int(final_pred > 0.5)



df = pd.read_csv('Model_Training_Data.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.index, df.label))
p_path = os.path.join('Pickle', 'alarm.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_pred = build_prediction('filename.wav')