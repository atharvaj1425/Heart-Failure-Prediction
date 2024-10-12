from django.shortcuts import render
from django import forms
import joblib
import numpy as np

# Load all models and scaler
knn_model = joblib.load('models3/knn_model01.pkl')
decision_tree_model = joblib.load('models3/decision_tree_model01.pkl')
random_forest_model = joblib.load('models3/random_forest_model01.pkl')
log_reg_model = joblib.load('models3/log_reg_model01.pkl')
scaler = joblib.load('models3/scaler.pkl')

# Define the form within the view for heart failure prediction
class HeartFailureForm(forms.Form):
    age = forms.IntegerField(label='Age', min_value=0, max_value=120)
    anaemia = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Anaemia')
    creatinine_phosphokinase = forms.IntegerField(label='Creatinine Phosphokinase (U/L)')
    diabetes = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Diabetes')
    ejection_fraction = forms.IntegerField(label='Ejection Fraction (%)', min_value=0, max_value=100)
    high_blood_pressure = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='High Blood Pressure')
    platelets = forms.FloatField(label='Platelets (kiloplatelets/mL)')
    serum_creatinine = forms.FloatField(label='Serum Creatinine (mg/dL)')
    serum_sodium = forms.FloatField(label='Serum Sodium (mEq/L)')
    sex = forms.ChoiceField(choices=[(0, 'Female'), (1, 'Male')], label='Sex')
    smoking = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Smoking')
    time = forms.IntegerField(label='Time (days since diagnosis)', min_value=0)

# View to handle form input and make predictions
def predict_heart_failure(request):
    if request.method == 'POST':
        form = HeartFailureForm(request.POST)
        if form.is_valid():
            # Extract form data and ensure the correct format
            data = [
                form.cleaned_data['age'],
                form.cleaned_data['anaemia'],  # Encoded 0 or 1 for anaemia
                form.cleaned_data['creatinine_phosphokinase'],
                form.cleaned_data['diabetes'],  # Encoded 0 or 1 for diabetes
                form.cleaned_data['ejection_fraction'],
                form.cleaned_data['high_blood_pressure'],  # Encoded 0 or 1 for high blood pressure
                form.cleaned_data['platelets'],
                form.cleaned_data['serum_creatinine'],
                form.cleaned_data['serum_sodium'],
                form.cleaned_data['sex'],  # Encoded 0 or 1 for sex
                form.cleaned_data['smoking'],  # Encoded 0 or 1 for smoking
                form.cleaned_data['time'],  # Days since diagnosis
            ]

            # Convert to numpy array and scale the data
            input_data = np.array([data])
            input_data_scaled = scaler.transform(input_data)

            # Get predictions from all models
            knn_prediction = knn_model.predict(input_data_scaled)[0]
            decision_tree_prediction = decision_tree_model.predict(input_data_scaled)[0]
            random_forest_prediction = random_forest_model.predict(input_data_scaled)[0]
            log_reg_prediction = log_reg_model.predict(input_data_scaled)[0]

            # Aggregate the predictions in a dictionary
            predictions = {
                'KNN': 'Heart Failure' if knn_prediction == 1 else 'No Heart Failure',
                'Decision Tree': 'Heart Failure' if decision_tree_prediction == 1 else 'No Heart Failure',
                'Random Forest': 'Heart Failure' if random_forest_prediction == 1 else 'No Heart Failure',
                'Logistic Regression': 'Heart Failure' if log_reg_prediction == 1 else 'No Heart Failure',
            }

            # Prepare data for the chart
            chart_labels = list(predictions.keys())
            chart_data = [1 if value == 'Heart Failure' else 0 for value in predictions.values()]

            # Render the results page with predictions and chart data
            return render(request, 'predict/results.html', {
                'predictions': predictions,
                'chart_labels': chart_labels,
                'chart_data': chart_data
            })

    else:
        form = HeartFailureForm()

    return render(request, 'predict/home.html', {'form': form})

# Optional results view if needed
def results_view(request):
    predictions = request.session.get('predictions', {})
    return render(request, 'predict/results.html', {'predictions': predictions})
