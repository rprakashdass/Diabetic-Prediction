# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:35:44 2023

@author: PRAKASH R
"""

#pickle -> to load saved model
#streamlit -> to create a webpage

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('C:/Users/PRAKASH R/Desktop/Machine Learning/diabetics/trained_model.sav', 'rb'))

# creating function to do prediction

def diabetic_predict(input_data):
        
    # input_data = (5,166,72,19,175,25.8,0.587,51)
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    print(prediction)
    
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #To give title for our page
    
    st.title('ML Web Application')
    
    # To get Input From the user

    Pregnancies	= st.text_input('Tell me the number of pregnencies you met')
    Glucose	= st.text_input('How much is your glucose level')
    BloodPressure = st.text_input('Tell me your Bloodpresuure rate')	
    SkinThickness	= st.text_input('Tell me the Thickness of your skin')
    Insulin	= st.text_input('whats your insulin level now')
    BMI	= st.text_input('What would be your BMI index ?')
    DiabetesPedigreeFunction	= st.text_input('Tell me your Diabetes Pedigree Function level')
    Age= st.text_input('How old you are?')
    
    
    #Main code for predicion
    output = ''
    
    #To crete a button for submission
    if st.button('test now'):
        output = diabetic_predict([Pregnancies,Glucose,	BloodPressure,SkinThickness, Insulin, BMI,	DiabetesPedigreeFunction, Age])
  
    
    st.success(output)
        
      
if __name__ == '__main__':
    main()