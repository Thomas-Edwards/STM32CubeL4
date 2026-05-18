import time
from datetime import datetime
import struct
from flask import Flask, render_template, request, jsonify
import threading
import os
import pickle
from tensorflow.keras.models import load_model
from predict import build_prediction 



app = Flask(__name__)

p_path = os.path.join('Pickle', 'alarm.p')
with open(p_path, 'rb') as handle:
    cfg = pickle.load(handle)
cfg.model_path = os.path.join('Model', 'alarm.h5')
if not hasattr(cfg, 'step'):
    cfg.step = 1600
model = load_model(cfg.model_path, compile=False)

latest_db_value = 0
latest_mag_value = 0
db_lock = threading.Lock()
alarm_time = None
classifier_alarm = False
last_alarm_time = 0
alarm_count = 0
SAVE_SAMPLES = True
SAMPLE_DIR = r'C:\Users\mkint\dev\Capstone\ThomasSTM32\Frontend\Dataset\Alarm'
sample_count = 0




SILENCE_MAG = 3000.0
MAX_MAG     = 100000.0
SMOOTHING   = 0.1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/alarm/set', methods=['POST'])
def set_alarm():
    global alarm_time
    data = request.json
    alarm_time = data.get('wakeup_time')
    print(f"Alarm set for: {alarm_time}")
    return jsonify({"message": "Alarm received"}), 200

@app.route('/api/audio', methods=['GET', 'POST'])
def audio_data():
    global latest_db_value, alarm_time, latest_mag_value

    if request.method == 'POST':
        raw = request.data

        if not raw:
            return jsonify({"error": "No data received"}), 400

        num_floats = len(raw) // 4
        floats = struct.unpack(f'{num_floats}f', raw[:num_floats*4])

        with db_lock:
            avg_mag = sum(floats) / len(floats)
            latest_mag_value = avg_mag
            normalized = (avg_mag - SILENCE_MAG) / (MAX_MAG - SILENCE_MAG)
            instant = max(0.0, min(100.0, normalized * 100.0))
            latest_db_value = (SMOOTHING * instant) + ((1 - SMOOTHING) * latest_db_value)
            print(f"avg_mag: {avg_mag:.0f}  ->  {latest_db_value:.1f}%")

        now = datetime.now().strftime("%H:%M")
        if alarm_time is not None and alarm_time == now:
            return "ALARM_ON", 200
        return "OK", 200

    with db_lock:
        now = datetime.now().strftime("%H:%M")
        alarm_active = alarm_time is not None and alarm_time == now
        return jsonify({
            "db": latest_db_value,
            "mag": latest_mag_value,
            "alarm": alarm_active or classifier_alarm
        }), 200


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global SILENCE_MAG, MAX_MAG
    data = request.json
    SILENCE_MAG = float(data.get('silence_mag', 3000.0))
    MAX_MAG     = float(data.get('max_mag', 100000.0))
    print(f"Calibration updated: SILENCE_MAG={SILENCE_MAG}, MAX_MAG={MAX_MAG}")
    return jsonify({"silence_mag": SILENCE_MAG, "max_mag": MAX_MAG}), 200

@app.route('/api/classify', methods=['POST'])
def classify():
    global classifier_alarm, last_alarm_time, alarm_count, sample_count
    raw = request.get_data()

    if SAVE_SAMPLES and raw:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        filepath = os.path.join(SAMPLE_DIR, f'ambient_{sample_count:04d}.bin')
        with open(filepath, 'wb') as f:
            f.write(raw)
        print(f"Saved ambient sample {sample_count}")
        sample_count += 1



    pred = build_prediction(raw, cfg, model)
    now = datetime.now().strftime("%H:%M")
    print(f"Final prediction: {pred}")

    if pred == 0:
        if time.time() - last_alarm_time < 5:
            alarm_count += 1
        else:
            alarm_count = 1
        last_alarm_time = time.time()
        print(f"Alarm count: {alarm_count}")
    else:
        alarm_count = 0


    if alarm_count >= 3 or (alarm_time is not None and alarm_time == now):
        classifier_alarm = True
        last_alarm_time = time.time()
        return "ALARM_ON", 200

    if time.time() - last_alarm_time > 3:
        classifier_alarm = False
        print("Classifier alarm reset")
    return "OK", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)