from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from models.arima_model import get_arima_forecast
from models.var_model import get_var_forecast
from models.event_analysis import get_event_impact

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load data
prices_data = pd.read_csv("C:\\Users\\Hasan\\Desktop\\data science folder\\BrentOilPrices.csv")
#events_data = pd.read_csv('data/events_data.csv')

# API to get historical price data
@app.route('/api/prices', methods=['GET'])
def get_prices():
    return jsonify(prices_data.to_dict(orient='records'))

# API for ARIMA model forecast
@app.route('/api/forecast/arima', methods=['GET'])
def forecast_arima():
    forecast = get_arima_forecast()
    return jsonify(forecast)

# API for model performance metrics
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics = {
        "MAE": 18.48,
        "RMSE": 22.15
    }
    return jsonify(metrics)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
