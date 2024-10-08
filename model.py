import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset without specifying dtype
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv"
data = pd.read_csv(url, low_memory=False)

# Replace '?' with NaN and convert columns to appropriate types
columns_to_convert = [
    'ForecastWindProduction', 
    'SystemLoadEA', 
    'SMPEA', 
    'ORKTemperature', 
    'ORKWindspeed', 
    'CO2Intensity', 
    'ActualWindProduction', 
    'SystemLoadEP2', 
    'SMPEP2'
]

# Replace '?' with NaN and convert to float
for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column].replace('?', np.nan), errors='coerce')

# Drop rows with NaN values if necessary
data.dropna(subset=columns_to_convert, inplace=True)

# Data Preprocessing (Adjust this based on your dataset)
X = data[['SystemLoadEA']]  # Features
y = data['SMPEA']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)
