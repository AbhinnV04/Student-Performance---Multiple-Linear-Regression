import streamlit as st 
import pickle
import numpy as np 

with open('model.pk1', 'rb') as file:
    exp_data = pickle.load(file)
 
reg = exp_data['model']

def show_predict_page():
    st.header("Student Performance Predictor")
    st.write('''
             #### ***(Multiple Linear Regression Model)***
             Dataset link - https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression''')
    hours_studied = st.slider('Hours of study', 0, 24, 5) 
    sleep_hours = st.slider('Hours of Sleep', 0, 24, 5) 
    Sample_Question_Papers = st.slider('Sample Question Papers Practiced', 0, 10, 4)
    prev_scores = st.number_input('Previous Score', min_value=0, max_value=100, step=1, value=50)
    extracurr = st.radio('Participation in Extracurricular Activities', ('Yes', 'No'))
   
    if extracurr == 'Yes':
        extracurr = 1.0
    else:
        extracurr = 0.0
   
    calculate = st.button('Calculate')
    
    if calculate:
        X = np.array([[hours_studied, prev_scores, extracurr, sleep_hours, Sample_Question_Papers]])
        X = X.astype(float)
        
        p_i = reg.predict(X)
        st.subheader(f"The estimated Performance Index is: {p_i[0]:.2f}")

show_predict_page()