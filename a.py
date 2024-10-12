import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode only the 'sex' column
df['sex'] = label_encoder.fit_transform(df['sex'])  # 0 for female, 1 for male

# Feature and target variables
X = df.drop(columns=['DEATH_EVENT'])  # Features
y = df['DEATH_EVENT']  # Target

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data (scale the features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate a model and store accuracy
def evaluate_model(model, model_name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, "K-Nearest Neighbors")

# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
evaluate_model(decision_tree, "Decision Tree")

# Random Forest
random_forest = RandomForestClassifier(random_state=42)
evaluate_model(random_forest, "Random Forest")

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
evaluate_model(log_reg, "Logistic Regression")

# Save models after training
joblib.dump(knn, 'models3/knn_model01.pkl')
joblib.dump(decision_tree, 'models3/decision_tree_model01.pkl')
joblib.dump(random_forest, 'models3/random_forest_model01.pkl')
joblib.dump(log_reg, 'models3/log_reg_model01.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler have been saved!")
