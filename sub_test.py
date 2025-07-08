from tensorflow.keras.models import load_model

model = load_model('model/usd_inr_lstm_model.h5', compile=False)
model.export('model/usd_inr_lstm_model_tf')  # saves in SavedModel format
