import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('/Users/samratvats/Downloads/HAM10000_metadata.csv')

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nDiagnosis value counts:")
print(df['dx'].value_counts())

# Define target: 'dx' (diagnosis)
# Encode labels
label_encoder = LabelEncoder()
df['dx_encoded'] = label_encoder.fit_transform(df['dx'])

# Select features: age, sex, localization
features = ['age', 'sex', 'localization']
X = df[features]
y = df['dx_encoded']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['sex', 'localization'])

# Handle missing values in age (if any)
X['age'].fillna(X['age'].median(), inplace=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print sample counts
print(f"\nTotal samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Standardize numerical features
scaler = StandardScaler()
X_train[['age']] = scaler.fit_transform(X_train[['age']])
X_test[['age']] = scaler.transform(X_test[['age']])

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Example testing module

print("EXAMPLE TESTING MODULE")


# Create some example test cases
example_cases = [
    {'age': 18, 'sex': 'male', 'localization': 'face'},
    {'age': 60, 'sex': 'female', 'localization': 'lower extremity'},
    {'age': 35, 'sex': 'male', 'localization': 'face'},
    {'age': 50, 'sex': 'female', 'localization': 'trunk'}
]

# Convert to DataFrame
example_df = pd.DataFrame(example_cases)

# Preprocess the examples like we did with the training data
example_processed = pd.get_dummies(example_df, columns=['sex', 'localization'])

# Ensure all columns from training are present in test examples
missing_cols = set(X_train.columns) - set(example_processed.columns)
for col in missing_cols:
    example_processed[col] = 0  # Add missing columns with 0 values

# Reorder columns to match training data
example_processed = example_processed[X_train.columns]

# Standardize age
example_processed[['age']] = scaler.transform(example_processed[['age']])

# Make predictions
example_predictions = clf.predict(example_processed)
example_probabilities = clf.predict_proba(example_processed)

# Display results
print("\nExample case predictions:")
for i, (case, pred, probs) in enumerate(zip(example_cases, example_predictions, example_probabilities)):
    diagnosis = label_encoder.inverse_transform([pred])[0]
    print(f"\nCase {i+1}: {case}")
    print(f"Predicted diagnosis: {diagnosis}")
    print("Probability distribution:")
    for j, cls in enumerate(label_encoder.classes_):
        print(f"  {cls}: {probs[j]:.4f}")

# Feature importance
feature_importance = clf.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)


print("FEATURE IMPORTANCE")

print(importance_df.head(10))  # Show top 10 most important features

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()

# Save model components for web app
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)