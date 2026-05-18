import numpy as np
from python_speech_features import mfcc
from keras.models import load_model
from python_speech_features import mfcc

def build_prediction(buffer, config, model, rate=16000):
    preds = []

    # convert raw bytes to numpy int32
    wav = np.frombuffer(buffer, dtype=np.int32)

    # convert to float
    wav = wav.astype(np.float32)

    # normalize audio amplitude
    wav = wav / 32768.0

    for i in range(0, wav.shape[0] - config.step, config.step):

        sample = wav[i:i + config.step]

        # skip silent chunks
        if np.abs(sample).mean() < 0.01:
            continue

        x = mfcc(sample, rate, numcep=config.nfeat,
                 nfilt=config.nfilt, nfft=config.nfft)

        # feature normalization
        x = (x - np.mean(x)) / np.std(x)

        # CNN input shape
        x = x.reshape(1, x.shape[0], x.shape[1], 1)
        pred = model.predict(x, verbose=0)[0][0]

        preds.append(pred)

    # handle empty predictions
    if len(preds) == 0:
        return 0

    final_pred = np.mean(preds)

    return int(final_pred > 0.5)
