import matplotlib.pyplot as plt
import numpy as np
import librosa
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import os
from scipy.io import wavfile
import pickle
from sklearn.metrics import accuracy_score

def build_prediction(audio_dir, rate=16000):
    y_true = []
    y_pred = []
    fd_prob = {}

    for rn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, rn))
        label = fn2class[rn]
        c = classes.index(label)

        sample = wav[:config.rate]

        # Pad if shorter than 1 second
        if len(sample) < config.rate:
            sample = np.pad(sample, (0, config.rate - len(sample)))

        X_sample = librosa.feature.melspectrogram(
            y=sample.astype(float), sr=rate, n_mels=128
        )
        X_sample = librosa.power_to_db(X_sample, ref=np.max)
        X_sample = (X_sample - np.mean(X_sample)) / np.std(X_sample)

        x = X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 1)

        y_hat = model.predict(x, verbose=0)
        fd_prob[rn] = y_hat[0][0]
        y_pred.append(1 if y_hat[0][0] > 0.5 else 0)
        y_true.append(c)

    return y_true, y_pred, fd_prob


df = pd.read_csv('Model_Training_Data.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.fname, df.label))

p_path = os.path.join('Pickle', 'alarm.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_prediction('Dataset/Original')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    
    df.at[i, 'filtered'] = 1 - y_prob
    df.at[i, 'not_filtered'] = y_prob

y_pred = ['not_filtered' if p > 0.5 else 'filtered' for p in y_probs]
df['y_pred'] = y_pred
df.to_csv('Predictions.csv', index=False)

# df = pd.read_csv('Predictions.csv')

# # Confidence = probability of the predicted class
# def get_confidence(row):
#     return row[row['y_pred']]

# df['confidence'] = df.apply(get_confidence, axis=1)
# df['correct'] = df['label'] == df['y_pred']

# x = np.arange(len(df))
# colors = ['#2ecc71' if c else '#e74c3c' for c in df['correct']]
# labels = [f.replace('sample_', '').replace('.wav', '') for f in df['fname']]

# fig, ax = plt.subplots(figsize=(12, 5))

# # Confidence bars (muted background)
# ax.bar(x, df['confidence'], color='#d0d0d0', width=0.6, zorder=1)

# # Dots on top
# ax.scatter(x, df['confidence'], color=colors, s=100, zorder=3, linewidths=1.5, edgecolors='white')

# # Threshold line
# ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
# ax.text(len(df) - 0.4, 0.52, 'threshold', fontsize=8, color='gray', va='bottom', ha='right')

# ax.set_title('Prediction confidence', fontsize=13, fontweight='bold')
# ax.set_xlabel('File')
# ax.set_ylabel('Confidence')
# ax.set_ylim(0, 1.05)
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

# # Annotate each dot with predicted class
# for i, row in df.iterrows():
#     ax.text(i, row['confidence'] + 0.03, row['y_pred'],
#             ha='center', va='bottom', fontsize=7, color='#444')

# # Legend
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
#            markersize=10, markeredgecolor='white', label='Correct'),
#     Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
#            markersize=10, markeredgecolor='white', label='Incorrect'),
# ]
# ax.legend(handles=legend_elements, fontsize=9)

# # Summary
# total = len(df)
# n_correct = df['correct'].sum()
# accuracy = n_correct / total * 100
# fig.text(0.5, -0.02,
#          f'Total: {total}  |  Correct: {n_correct}  |  Accuracy: {accuracy:.1f}%  |  Avg confidence: {df["confidence"].mean():.3f}',
#          ha='center', fontsize=10, color='#555')

# plt.tight_layout()
# plt.savefig('confidence_plot.png', dpi=150, bbox_inches='tight')
# print('Saved confidence_plot.png')
# plt.show()