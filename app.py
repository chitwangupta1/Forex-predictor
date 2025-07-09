import pandas as pd
import numpy as np
from flask import Flask, render_template, request
# from keras.models import load_model
from keras.models import load_model
import os

# Load model and dataset
model = load_model('model/usd_inr_lstm_model.h5', compile=False)
df = pd.read_csv('usd_to_inr_new.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.set_index('Date', inplace=True)
df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='D'), method='ffill')

# Technical indicators
df['SMA_7'] = df['Value'].rolling(window=7).mean()
df['EMA_7'] = df['Value'].ewm(span=7, adjust=False).mean()

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI_14'] = compute_RSI(df['Value'])
df.bfill(inplace=True)

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    chart_path = None

    if request.method == 'POST':
        try:
            input_date = pd.to_datetime(request.form['date'])
            if input_date < df.index.min():
                return render_template('index.html', predictions=["Date is too far in the past."], chart=None)

            window_size = 3
            pred_dates = []
            values = []

            temp_df = df.copy()
            today = df.index.max()

            # CASE 1: Show actuals for selected and next 3 days
            if input_date <= today:
                for i in range(4):
                    current_day = input_date + pd.Timedelta(days=i)
                    if current_day in temp_df.index:
                        val = temp_df.loc[current_day, 'Value']
                        predictions.append(f"{current_day.strftime('%Y-%m-%d')} (Actual): ₹{val:.4f}")
                        pred_dates.append(current_day)
                        values.append(val)
                    else:
                        predictions.append(f"{current_day.strftime('%Y-%m-%d')}: Data not available")

            # CASE 2: Predict future values
            else:
                last_known = temp_df.index.max()

                while input_date > last_known:
                    recent_window = pd.date_range(end=last_known, periods=window_size)
                    if not all(d in temp_df.index for d in recent_window):
                        predictions.append("Not enough data to generate prediction.")
                        return render_template('index.html', predictions=predictions, chart=None)

                    input_data = temp_df.loc[recent_window][['Value', 'SMA_7', 'EMA_7', 'RSI_14']].values
                    input_data = input_data.reshape((1, window_size, 4))
                    pred = model.predict(input_data)[0][0]
                    next_day = last_known + pd.Timedelta(days=1)

                    # Add prediction & update indicators
                    temp_df.loc[next_day, 'Value'] = pred
                    temp_df['SMA_7'] = temp_df['Value'].rolling(7).mean()
                    temp_df['EMA_7'] = temp_df['Value'].ewm(span=7, adjust=False).mean()
                    temp_df['RSI_14'] = compute_RSI(temp_df['Value'])
                    temp_df.bfill(inplace=True)

                    last_known = next_day

                # Predict from input date onward
                for i in range(4):
                    date_range = pd.date_range(end=input_date - pd.Timedelta(days=1), periods=window_size)
                    if not all(d in temp_df.index for d in date_range):
                        predictions.append(f"{input_date.strftime('%Y-%m-%d')}: Not enough data to predict.")
                        break

                    input_data = temp_df.loc[date_range][['Value', 'SMA_7', 'EMA_7', 'RSI_14']].values
                    input_data = input_data.reshape((1, window_size, 4))
                    pred = model.predict(input_data)[0][0]

                    temp_df.loc[input_date, 'Value'] = pred
                    temp_df['SMA_7'] = temp_df['Value'].rolling(7).mean()
                    temp_df['EMA_7'] = temp_df['Value'].ewm(span=7, adjust=False).mean()
                    temp_df['RSI_14'] = compute_RSI(temp_df['Value'])
                    temp_df.bfill(inplace=True)

                    predictions.append(f"{input_date.strftime('%Y-%m-%d')} (Predicted): ₹{pred:.4f}")
                    pred_dates.append(input_date)
                    values.append(pred)
                    input_date += pd.Timedelta(days=1)

        except Exception as e:
            predictions.append(f"Error: {e}")

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
