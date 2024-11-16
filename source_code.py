import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to select a CSV file
def select_file():
    Tk().withdraw()  # Prevent the root window from appearing
    file_path = askopenfilename(title="Select the CSV file", filetypes=[("CSV files", "*.csv")])
    return file_path

# Prompt user to select the file
file_path = select_file()

# Read the dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path and try again.")
    exit(1)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Extract relevant columns for features and target variable
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Example prediction
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]  # New data point
predicted_price = model.predict(new_data)
print("Predicted Price for new data:", predicted_price[0])
