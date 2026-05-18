import numpy as np
from python_speech_features import mfcc
from keras.models import load_model
from python_speech_features import mfcc

def build_prediction(buffer, config, model, rate=16000):
    wav = np.frombuffer(buffer, dtype=np.int32)
    wav = wav.astype(np.float32) / 32768.0

    if np.abs(wav).mean() < 0.000001:
        return 0

    x = mfcc(wav, rate, numcep=32, nfilt=40, nfft=512)
    x = (x - np.mean(x)) / np.std(x)
    
    # pad or trim to 128 frames
    if x.shape[0] < 128:
        x = np.pad(x, ((0, 128 - x.shape[0]), (0, 0)))
    else:
        x = x[:128]
    
    x = x.reshape(1, x.shape[0], x.shape[1], 1)
    pred = model.predict(x, verbose=0)[0][0]
    
    print(f"Raw score: {pred:.3f} -> Prediction: {int(pred > 0.5)}")
    return int(pred > 0.5)