from datetime import datetime

from flask import Flask, render_template, request, jsonify
import threading


app = Flask(__name__)

# Global variable to store the latest dB reading
latest_db_value = 0
db_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/alarm/set', methods=['POST']) 
def set_alarm():
    global alarm_time
    data = request.json  # This gets the { wakeup_time: val } from your JS
    alarm_time = data.get('wakeup_time')
    print(f"Alarm set for: {alarm_time}")
    return jsonify({"message": "Alarm received"}), 200


@app.route('/api/audio', methods=['GET', 'POST'])
def audio_data():
    """Receive audio data from STM32 or serve latest reading to frontend"""
    global latest_db_value, alarm_time

    if request.method == 'POST':
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
            return jsonify({"error": "Missing wakeup_time in request body"}), 400
        
        floats_1024 = json_data['data']

        with db_lock:
            #Simple RMS-to-dB approximation for the UI
            avg_mag = sum(floats_1024) / len(floats_1024)
            latest_db_value = 20 * (avg_mag + 1) # Scaling for UI

        # CHECK ALARM TRIGGER
        now = datetime.now().strftime("%H:%M")
        if alarm_time == now:
            return "ALARM_ON", 200
            
        return "OK", 200
    
    # Frontend GET request
    with db_lock:
        return jsonify({"db": latest_db_value}), 200

    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5000)
