

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the file path to your Excel file
df = pd.read_csv('transformed_dataset.csv')

# Convert 'InvoiceDate' column to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# EDA: Plot sales over time
plt.figure(figsize=(12, 6))
plt.plot(df['InvoiceDate'], df['Quantity'])
plt.title('Sales Over Time')
plt.xlabel('Invoice Date')
plt.ylabel('Quantity Sold')
plt.show()

# Feature Engineering: Extract year, month, day, and weekday from 'InvoiceDate'
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Weekday'] = df['InvoiceDate'].dt.weekday

# Model Building
# Split data into train and test sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Define and fit ARIMA model
model = ARIMA(train['Quantity'], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Validation
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['Quantity'], forecast))
print('RMSE:', rmse)

# Plot actual vs forecasted sales
plt.figure(figsize=(12, 6))
plt.plot(test['InvoiceDate'], test['Quantity'], label='Actual')
plt.plot(test['InvoiceDate'], forecast, label='Forecast')
plt.title('Actual vs Forecasted Sales')
plt.xlabel('Invoice Date')
plt.ylabel('Quantity Sold')
plt.legend()
plt.show()
