import pandas as pd
import numpy as np
import joblib

# Load the saved models and scaler from models 3
knn = joblib.load('models3/knn_model01.pkl')
decision_tree = joblib.load('models3/decision_tree_model01.pkl')
random_forest = joblib.load('models3/random_forest_model01.pkl')
log_reg = joblib.load('models3/log_reg_model01.pkl')
scaler = joblib.load('models3/scaler.pkl')

# Function to make a prediction using a model
def predict_heart_failure(model, user_data):
    user_data_df = pd.DataFrame([user_data], columns=[
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 
        'sex', 'smoking', 'time'
    ])  # Create DataFrame with feature names

    user_data_scaled = scaler.transform(user_data_df)  # Scale the new input using the same scaler
    prediction = model.predict(user_data_scaled)
    return 'Heart Failure' if prediction == 1 else 'No Heart Failure'

# Sample input (12 input features)
# Example: 
# - Age: 45
# - Anaemia: 0 (no)
# - Creatinine Phosphokinase: 100 (mcg/L)
# - Diabetes: 0 (no)
# - Ejection Fraction: 60%
# - High Blood Pressure: 0 (no)
# - Platelets: 300000 (kiloplatelets/mL)
# - Serum Creatinine: 1.0 (mg/dL)
# - Serum Sodium: 140 (mEq/L)
# - Sex: 1 (male)
# - Smoking: 0 (non-smoker)
# - Time: 30 (days)
new_input =[75, 1, 300, 1, 25, 1, 150000, 2.5, 130, 1, 1, 10]



# Predicting heart failure using all models
print("\nPrediction for new input using different models:")
models = {'KNN': knn, 'Decision Tree': decision_tree, 'Random Forest': random_forest, 'Logistic Regression': log_reg}

for model_name, model in models.items():
    prediction = predict_heart_failure(model, new_input)
    print(f"{model_name}: {prediction}")
