
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read the CSV file into a DataFrame
file_path = '/content/transformed_datasets.csv'  # Adjust the file path accordingly
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
df.head()

df.describe()
df.isnull().sum()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is already loaded as df
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is already loaded as df
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is already loaded as df
# Convert 'InvoiceDate' column to datetime format with specified format and handle errors
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m-%d-%Y %H:%M', errors='coerce')

# Drop rows with missing or incorrect dates
df = df.dropna(subset=['InvoiceDate'])

# Extract month and year from 'InvoiceDate'
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

# Group by 'YearMonth' and sum the 'Price' column
monthly_sales = df.groupby('YearMonth')['Price'].sum()

# Display the monthly sales
print(monthly_sales)

# Visualize monthly sales
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', marker='o', color='b')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Which invoices had the most items?
inv_counts = df['Invoice'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(x=inv_counts.index, y=inv_counts.values, palette=sns.color_palette("BuGn_d"))
plt.ylabel("Counts")
plt.title("Which invoices had the most items?")
plt.xticks(rotation=90)
plt.show()

df[df['Invoice'].str.startswith('C')].describe()



import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Assuming your DataFrame is already loaded as df and you have monthly sales data
# Convert 'YearMonth' column to datetime format
monthly_sales.index = pd.to_datetime(monthly_sales.index.to_timestamp())

# Fit ARIMA model
model = ARIMA(monthly_sales, order=(5, 1, 0))  # Example order, you may need to tune this
model_fit = model.fit()

# Forecast future sales
forecast_steps = 12  # Example: forecast sales for the next 12 months
forecast = model_fit.forecast(steps=forecast_steps)

# Visualize the forecast
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, label='Actual Sales')
plt.plot(forecast.index, forecast.values, label='Forecasted Sales', color='red')
plt.title('Forecasted Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Print the forecasted sales
print("Forecasted Sales:")
print(forecast)

