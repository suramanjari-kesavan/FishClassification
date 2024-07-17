import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv('Fish.csv')

# Display first few rows and check for missing values
print(df.head())
print(df.isnull().sum())

# Encode the species column
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

# Define features and target variable
X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Save the trained model
with open('fish_species_model.pkl', 'wb') as file:
    pickle.dump(model, file)
