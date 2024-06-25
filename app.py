import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Random Forest modelini tanımla ve eğit
rf_model = RandomForestClassifier(random_state=42)

# Model eğitimi için örnek veri (kendi modelinizi eğitmek için gerçek verileri kullanmalısınız)
# Bu örnekte modeli basit bir örnek veri seti ile eğitiyoruz.
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

rf_model.fit(X, y)

# Kullanıcı girişi arayüzü
st.title('Diabetes Prediction')

st.sidebar.header('User Input Features')
def user_input_features():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=140, value=70)
    skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree_function = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Model tahmini
prediction = rf_model.predict(input_df)
prediction_proba = rf_model.predict_proba(input_df)

st.subheader('Prediction')
outcome = 'Diabetic' if prediction[0] == 1 else 'Healthy'
st.write(outcome)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("""
### Notes:
* This application predicts diabetes based on user input values.
* The model is trained using a simple example dataset.
""")
