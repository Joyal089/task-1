import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the train and test datasets
train_data = pd.read_csv(r'C:\Users\joyal\Downloads\train.csv')
test_data = pd.read_csv(r'C:\Users\joyal\Downloads\test.csv')

# Extract relevant columns
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'

# Prepare the training data
X = train_data[features]
y = train_data[target]

# Split the training data for validation purposes
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate the Mean Squared Error (MSE) on the validation set
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error: {mse}')

# Prepare the test data
X_test = test_data[features]

# Predict house prices for the test dataset
test_predictions = model.predict(X_test)

# Display the first few predictions
print('First few predictions on test data:')
print(test_predictions[:5])
