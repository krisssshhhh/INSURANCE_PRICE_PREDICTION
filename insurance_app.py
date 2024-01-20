# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:12:39 2024

@author: dk694
"""

import numpy as np
import pickle
import streamlit as st
model_path = 'medical_insurance_cost_predictor.sav'
loaded_model = pickle.load(open(model_path, 'rb'))

# creating a function for Prediction
def medical_insurance_cost_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

def main():
    
    st.title('Medical Insurance Prediction Web App')
    
    age = st.text_input('Age')
    sex = st.text_input('Sex: 0 -> Female, 1 -> Male')
    bmi = st.text_input('Body Mass Index')
    children = st.text_input('Number of Children')
    smoker = st.text_input('Smoker: 0 -> No, 1 -> Yes')
    region = st.text_input('Region of Living: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest')
    
    diagnosis = ''
    if st.button('Predicted Medical Insurance Cost: '):
        input_data = [age, sex, bmi, children, smoker, region]
        diagnosis = medical_insurance_cost_prediction(input_data)
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
