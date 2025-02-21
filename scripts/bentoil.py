import pandas as pd

# Load the dataset 
df = pd.read_csv("C:\\Users\\Hasan\\Desktop\\data science folder\\BrentOilPrices.csv")

# Display the first few rows of the data to understand its structure
print(df.head())

# Convert the 'date' column to datetime (if it's in string format)
df['Date'] = pd.to_datetime(df['Date'])

# Set the date as the index (optional)
df.set_index('Date', inplace=True)
# Calculate the daily percentage change in Brent oil prices
df['price_change'] = df['Price'].pct_change() * 100

# Find days with significant price changes (for example, more than 5% change)
significant_changes = df[abs(df['price_change']) > 5]

# Display the rows with significant price changes
print(significant_changes)
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Check the column names to ensure there's a 'Price' column and if there's no 'Date'
print(df.columns)

# Step 3: If there is no 'Date' column, create a 'Date' column (assuming the dataset starts on 20-May-87)
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start='1987-05-20', periods=len(df), freq='D')

# Step 4: Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Step 5: Clean the 'Price' column to ensure it's numeric (if not already)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Step 6: Set the 'Date' column as the index (optional for time series analysis)
df.set_index('Date', inplace=True)

# Step 7: Plotting the price data
plt.figure(figsize=(12, 6))  # Adjust figure size (width=12, height=6)
plt.plot(df.index, df['Price'], label='Brent Oil Price', color='b')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Brent Oil Prices Over Time')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Ensure layout doesn't get cut off
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series data to analyze trend, seasonality, and residuals
result = seasonal_decompose(df['Price'], model='multiplicative', period=365)

# Plot the decomposition results
result.plot()
plt.show()
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import numpy as np

# Split the data into training and testing sets (80% training, 20% testing)
train, test = train_test_split(df['Price'], test_size=0.2, shuffle=False)

# Fit the ARIMA model (p, d, q values need to be tuned)
# For simplicity, we'll start with (1, 1, 1) - you can optimize this using grid search or auto_arima
model = ARIMA(train, order=(1, 1, 1))
fitted_model = model.fit()

# Make predictions for the test set
forecast = fitted_model.forecast(steps=len(test))

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual Prices', color='blue')
plt.plot(test.index, forecast, label='Predicted Prices', color='red')
plt.title('ARIMA Model: Actual vs Predicted Brent Oil Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Evaluate model performance (Mean Absolute Error)
mae = np.mean(np.abs(forecast - test))
print(f'Mean Absolute Error: {mae}')
from arch import arch_model

# GARCH model requires returns, so first calculate the returns of the oil prices
df['returns'] = df['Price'].pct_change().dropna()

# Fit a GARCH model to the returns
model_garch = arch_model(df['returns'].dropna(), vol='Garch', p=1, q=1)
garch_fit = model_garch.fit()

# Forecast volatility for the next 5 days
forecast_volatility = garch_fit.forecast(horizon=5)

# Plot the volatility forecast
plt.figure(figsize=(12, 6))
plt.plot(forecast_volatility.variance.values[-1, :], label='Forecasted Volatility', color='red')
plt.title('GARCH Model: Forecasted Volatility for Next 5 Days')
plt.xlabel('Days Ahead')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# Print the forecasted volatility values
print(forecast_volatility.variance.values[-1, :])
# Summary statistics of the price data
summary = df['Price'].describe()

# Print the summary
print("Summary Statistics for Brent Oil Prices:")
print(summary)


# Example: Report insights
print("\nInsights:")
print("1. The data shows significant volatility in Brent oil prices due to global supply-demand shifts.")
print("2. ARIMA model provides a reasonable prediction of future prices, but there may be inaccuracies in times of geopolitical instability.")
print("3. The GARCH model captures periods of high volatility during market shocks, providing insights into future price risks.")
# Adapt the Knowledge from Task 1.1 to Analyze Brent Oil Prices:
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Check if 'Date' column exists in the dataset
if 'Date' not in df.columns:
    # Create a synthetic Date range starting from '01-Jan-2000'
    df['Date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
    print("Date column was missing. Created synthetic dates.")
else:
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Decompose the time series data (Price) to analyze trend, seasonality, and residuals
result = seasonal_decompose(df['Price'], model='multiplicative', period=365)

# Plot the decomposition components
result.plot()
plt.show()
from statsmodels.tsa.stattools import adfuller

# Check for stationarity
adf_test = adfuller(df['Price'].dropna())
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# If p-value < 0.05, the series is stationary. If not, difference the series.
#Utilize Additional Statistical and Econometric Models as Needed:
import numpy as np

# Simulate GDP data (for demonstration purposes)
np.random.seed(42)  # For reproducibility
df['GDP'] = np.random.rand(len(df)) * 1000  # Random GDP values

# Check the updated dataframe columns
print(df.columns)

# Now apply the VAR model with the simulated GDP data
from statsmodels.tsa.api import VAR

model = VAR(df[['Price', 'GDP']])  # Use 'Price' and the simulated 'GDP'
results = model.fit(maxlags=15, ic='aic')
print(results.summary())
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare the data (scale the prices)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

# Prepare the dataset for LSTM
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Predict future prices
predicted_prices = model.predict(X)
