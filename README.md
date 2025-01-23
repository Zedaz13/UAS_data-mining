# UAS_data-mining
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load data, specifying the encoding
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data mining/UAS/Global YouTube Statistics.csv', encoding='latin-1')

# Print the available columns to verify the column names
print("Available columns:", data.columns)

# Preprocessing
data.dropna(inplace=True)  # Remove missing values

# Adjust the column names based on the output of data.columns
X = data[['subscribers_for_last_30_days', 'video_views_rank', 'category']]  # Features
y = data['video views']  # Target

# --- One-Hot Encoding for 'category' column ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the training data
encoded_train = encoder.fit_transform(X_train[['category']])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(['category']), index=X_train.index)

# Transform the test data using the same encoder
encoded_test = encoder.transform(X_test[['category']])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(['category']), index=X_test.index)

# Drop the original 'category' column and concatenate the encoded features
X_train = X_train.drop('category', axis=1).join(encoded_train_df)
X_test = X_test.drop('category', axis=1).join(encoded_test_df)
# --- End of One-Hot Encoding ---

# Modeling
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
