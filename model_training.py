import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Create synthetic weather dataset
np.random.seed(42)
n_samples = 1000

temperature = np.random.uniform(10, 35, n_samples)
humidity = np.random.uniform(30, 100, n_samples)
wind_speed = np.random.uniform(0, 25, n_samples)
rainfall = (-0.6 * temperature + 0.8 * humidity + 0.3 * wind_speed + 
           np.random.normal(0, 5, n_samples))

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'rainfall': rainfall
})

# Split data into features and target
X = df[['temperature', 'humidity', 'wind_speed']]
y = df['rainfall']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Create directory if not exists
os.makedirs('models', exist_ok=True)

# Save model and scaler
with open('models/weather_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Model trained and saved successfully!")
