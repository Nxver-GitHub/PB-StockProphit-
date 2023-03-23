from polygon import RESTClient
import config
import json
from typing import cast
from urllib3 import HTTPResponse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the RESTClient with the API key
client = RESTClient(config.API_KEY)

# Set end date to today and start date from 1 year ago
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)

# Retrieve stock data from company using daily aggregation for specified date range
aggs = cast(
    HTTPResponse,
    client.get_aggs(
        'AAPL',
        1,
        'day',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        raw=True
    ),
)

# Load the data into a Pandas DataFrame and convert timestamps to dates
data = json.loads(aggs.data)['results']
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['t'] / 1000, unit='s').dt.date

# Select features and target variable for the model
X = df[['o', 'h', 'l', 'v']]
y = df['c']

# Visualize the closing prices over time
plt.plot(df['date'], df['c'])
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

