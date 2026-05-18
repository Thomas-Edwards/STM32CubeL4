from tensorflow.keras.models import load_model
model = load_model('Model/alarm.keras')
model.save('Model/alarm.h5')
print("Done")
