from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('final_rf_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        data = [
            float(request.form['age']),
            float(request.form['annual_income']),
            float(request.form['monthly_inhand_salary']),
            int(request.form['num_bank_accounts']),
            int(request.form['num_credit_cards']),
            float(request.form['interest_rate']),
            int(request.form['num_of_loan']),
            int(request.form['delay_from_due_date']),
            int(request.form['num_of_delayed_payment']),
            float(request.form['changed_credit_limit']),
            int(request.form['num_credit_inquiries']),
            request.form['credit_mix'],
            float(request.form['outstanding_debt']),
            float(request.form['credit_utilization_ratio']),
            float(request.form['credit_history_age']),
            float(request.form['total_debt'])
        ]
        
        # One-hot encoding for credit mix
        credit_mix = data[11]
        if credit_mix == 'Poor':
            data = data[:11] + [1, 0, 0] + data[12:]
        elif credit_mix == 'Standard':
            data = data[:11] + [0, 1, 0] + data[12:]
        else:  # Good
            data = data[:11] + [0, 0, 1] + data[12:]

        # Convert to numpy array
        data = np.array(data).reshape(1, -1)
        
        # Predict
        prediction = model.predict(data)[0]
        
        return render_template('result.html', prediction=prediction)
    return 'Method Not Allowed', 405

if __name__ == '__main__':
    app.run(debug=True)