from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("Diabetes_Model.pkl")
scaler = joblib.load("Scaler.pkl")

@app.route('/api/diabetes', methods=['POST'])
def diabetes():
    # รับค่าจาก request
    pregnancies = float(request.form.get('pregnancies'))
    glucose = float(request.form.get('glucose'))
    blood_pressure = float(request.form.get('blood_pressure'))
    skin_thickness = float(request.form.get('skin_thickness'))
    insulin = float(request.form.get('insulin'))
    bmi = float(request.form.get('bmi'))
    age = float(request.form.get('age'))
    
    # Prepare the input for the model
    x = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age]])
    
    # Scale the input data
    x_scaled = scaler.transform(x)

    # Predict using the model
    prediction = model.predict(x_scaled)
    if int(prediction[0] == 0):
        return {'ผลการคาดการ': 'คุณอาจไม่เป็นโรคเบาหวาน'}
    else:
        return {'ผลการคาดการ': 'คุณอาจเป็นโรคเบาหวาน'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
