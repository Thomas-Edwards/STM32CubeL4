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
    global latest_db_value

    # STM32 is sending data
    db_value = request.args.get('db', type=float)
    if db_value is not None:
        with db_lock:
            latest_db_value = db_value
        # trigger the physical alarm
        now = datetime.now().strftime("%H:%M")
        if alarm_time == now:
            # This string triggers the strstr() check in your main.c
            return "ALARM_ON", 200 
            
        return "OK", 200
    
    # Frontend is requesting data
    with db_lock:
        return jsonify({"db": latest_db_value}), 200
    


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
