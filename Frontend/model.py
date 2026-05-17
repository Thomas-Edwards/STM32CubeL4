import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from scipy.io import wavfile
import librosa
from python_speech_features import mfcc, logfbank
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input
from sklearn.utils.class_weight import compute_class_weight
import pickle
from keras.callbacks import ModelCheckpoint
from STM32CubeL4.Frontend.cfg import Config

def check_data():
    if os.path.isfile(config.p_path):
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('Dataset/Original/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat, nfilt=config.nfilt,
                        nfft=config.nfft).T
        X_sample = (X_sample - np.mean(X_sample)) / np.std(X_sample)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))

    X, y = np.array(X), np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    y = np.array(y)
    config.data = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X, y

# Read the csv with the training data
df = pd.read_csv('Model_Training_Data.csv')
df.set_index('fname', inplace=True)

# Calculates the length in sec of each of the samples
for f in df.index:
    rate, signal = wavfile.read('Dataset/Original/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

def get_conv_model():
    model = Sequential([Input(shape=input_shape)])

    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                    padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                    padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['acc'])
    return model

config = Config()

X, y = build_rand_feat()
input_shape = (X.shape[1], X.shape[2], 1)
model = get_conv_model()

class_weight = dict(enumerate(compute_class_weight(class_weight='balanced',
                                    classes=np.unique(y), y=y)))

checkpoint = ModelCheckpoint(config.model_path,
                             monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False)

model.fit(X, y, epochs=50, batch_size=32, shuffle=True,
          class_weight=class_weight, validation_split=0.1,
          callbacks=[checkpoint])

model.save(config.model_path)

