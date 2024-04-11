import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('autism_model.sav','rb'))

def check(input_data):
    array_input = np.array(input_data)
    reshaped_input = array_input.reshape(1,-1)
    prediction = loaded_model.predict(reshaped_input)
    return format(prediction[0])

def main():
    st.title("Autism Prediction")

    age = st.number_input("Age")
    Gender = st.number_input("Gender")  # Changed variable name from gender to Gender
    ethnicity = st.number_input("Ethnicity")
    jaundice = st.number_input("Jaundice")
    autism = st.number_input("Autism")
    country_of_res = st.number_input("Country_of_res")
    A1_Score = st.number_input("A1_Score")
    A2_Score = st.number_input("A2_Score")
    A3_Score = st.number_input("A3_Score")
    A4_Score = st.number_input("A4_Score")
    A5_Score = st.number_input("A5_Score")
    A6_Score = st.number_input("A6_Score")
    A7_Score = st.number_input("A7_Score")
    A8_Score = st.number_input("A8_Score")
    A9_Score = st.number_input("A9_Score")
    A10_Score = st.number_input("A10_Score")

    pred = ""
    if st.button("Click Here for Autism Prediction"):
        pred = check([age, Gender, ethnicity, jaundice, autism, country_of_res, A2_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score])

    st.success(f"Your Autism prediction is {pred}")

if __name__=='__main__':
    main()
