import numpy as np
from scipy.io import wavfile
import os

def convert_folder(bin_dir, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(bin_dir) if f.endswith('.bin')]
    rows = []
    for f in files:
        path = os.path.join(bin_dir, f)
        with open(path, 'rb') as handle:
            raw = handle.read()
        samples = np.frombuffer(raw, dtype=np.int32)
        wav = samples.astype(np.int16)
        wav_name = f.replace('.bin', '.wav')
        out_path = os.path.join(out_dir, wav_name)
        wavfile.write(out_path, 16000, wav)
        rows.append(f"{wav_name},{label}")
        print(f"Converted {f}")
    return rows

rows = []
rows += convert_folder('Dataset/Alarm', 'Dataset/Original/Alarm', 'filtered')
rows += convert_folder('Dataset/Ambient', 'Dataset/Original/Ambient', 'not_filtered')

with open('Model_Training_Data_New.csv', 'w') as f:
    f.write('fname,label\n')
    f.write('\n'.join(rows))

print(f"Done - {len(rows)} samples")
