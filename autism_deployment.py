import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import os
import warnings

warnings.filterwarnings("ignore")

# Load the model
loaded_model = pk.load(open('autism_model.sav', 'rb'))

# Country mapping dictionary
country_mapping = {'Jordan': 0, 'United States': 1, 'Egypt': 2, 'United Kingdom': 3, 'Bahrain': 4, 'Austria': 5, 'Kuwait': 6, 
                   'United Arab Emirates': 7, 'Europe': 8, 'Malta': 9, 'Bulgaria': 10, 'South Africa': 11, 'India': 12, 
                   'Afghanistan': 13, 'Georgia': 14, 'New Zealand': 15, 'Syria': 16, 'Iraq': 17, 'Australia': 18, 
                   'Saudi Arabia': 19, 'Armenia': 20, 'Turkey': 21, 'Pakistan': 22, 'Canada': 23, 'Oman': 24, 'Brazil': 25, 
                   'South Korea': 26, 'Costa Rica': 27, 'Sweden': 28, 'Philippines': 29, 'Malaysia': 30, 'Argentina': 31, 
                   'Japan': 32, 'Bangladesh': 33, 'Qatar': 34, 'Ireland': 35, 'Romania': 36, 'Netherlands': 37, 'Lebanon': 38, 
                   'Germany': 39, 'Latvia': 40, 'Russia': 41, 'Italy': 42, 'China': 43, 'Nigeria': 44, 'U.S. Outlying Islands': 45, 
                   'Nepal': 46, 'Mexico': 47, 'Isle of Man': 48, 'Libya': 49, 'Ghana': 50, 'Bhutan': 51, 'American Samoa': 52, 
                   'Albania': 53, 'Belgium': 54, 'Azerbaijan': 55, 'Croatia': 56, 'France': 57, 'Indonesia': 58, 'Greenland': 59, 
                   'Bahamas': 60, 'Viet Nam': 61, 'Comoros': 62, 'Portugal': 63, 'Finland': 64, 'Norway': 65, 'Anguilla': 66, 
                   'Spain': 67, 'Burundi': 68, 'Chile': 69, 'Tonga': 70, 'Sri Lanka': 71, 'Sierra Leone': 72, 'Ethiopia': 73, 
                   'Iran': 74, 'Iceland': 75, 'Nicaragua': 76, 'Hong Kong': 77, 'Ukraine': 78, 'Kazakhstan': 79, 'Uruguay': 80, 
                   'Serbia': 81, 'Ecuador': 82, 'Niger': 83, 'Bolivia': 84, 'Aruba': 85, 'Angola': 86, 'Czech Republic': 87, 
                   'Cyprus': 88}

# Function to convert categorical variables to numerical
def convert_to_numerical(value, categories):
    return categories.index(value)

def check(input_data):
    array_input = np.array(input_data)
    reshaped_input = array_input.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_input)
    return format(prediction[0])

def main():
    st.markdown("<h1 style='text-decoration: underline; color: blue;'>Autism Prediction using ML</h1>", unsafe_allow_html=True)

    # Adding a background image
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://i.postimg.cc/0Q3bdMDw/autism-background.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Enhancing layout with visually appealing columns
    #st.markdown("<h2 style='color: #4CAF50;'>Please fill in the details below:</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("**Age**")

    with col2:
        gender = st.selectbox("**Gender**", ['Male', 'Female'])

    with col3:
        ethnicity = st.selectbox("**Ethnicity**", ['Asian', 'Black', 'White-European', 'Hispanic', 'Latino', 
                                               'Middle-Eastern', 'Others', 'Pasifika', 'South Asian', 'Turkish'])

    with col1:
        jaundice = st.selectbox("**Jaundice**", ['Yes', 'No'])

    with col2:
        autism = st.selectbox("**Autism**", ['Yes', 'No'])

    with col3:
        country_of_res = st.selectbox("**Country of Residence**", list(country_mapping.keys()))  # Changed to selectbox for consistency

    with col1:
        A1_Score = st.number_input("**A1_Score**")

    with col2:
        A2_Score = st.number_input("**A2_Score**")

    with col3:
        A3_Score = st.number_input("**A3_Score**")

    with col1:
        A4_Score = st.number_input("**A4_Score**")

    with col2:
        A5_Score = st.number_input("**A5_Score**")

    with col3:
        A6_Score = st.number_input("**A6_Score**")

    with col1:
        A7_Score = st.number_input("**A7_Score**")

    with col2:
        A8_Score = st.number_input("**A8_Score**")

    with col3:
        A9_Score = st.number_input("**A9_Score**")

    with col2:
        A10_Score = st.number_input("**A10_Score**")

    pred = ""
    if st.button("**Click Here for Autism Prediction**"):
        # Convert categorical variables to numerical
        gender = convert_to_numerical(gender, ['Male', 'Female'])
        ethnicity = convert_to_numerical(ethnicity, ['Asian', 'Black', 'White-European', 'Hispanic', 'Latino', 
                                                    'Middle-Eastern', 'Others', 'Pasifika', 'South Asian', 'Turkish'])
        jaundice = convert_to_numerical(jaundice, ['Yes', 'No'])
        autism = convert_to_numerical(autism, ['Yes', 'No'])
        country_of_res = country_mapping[country_of_res]

        # Create the input array for prediction
        input_data = [age, gender, ethnicity, jaundice, autism, country_of_res, A1_Score, A2_Score, A3_Score, 
                      A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score]

        pred = check(input_data)

        if pred == '1':
            autism_diagnosis = 'The person has an Autism Problem'
            result_color = "red"
        else:
            autism_diagnosis = 'The person does not have an Autism Problem'
            result_color = "green"

        st.markdown(f"<h3 style='color: {result_color};'><b>{autism_diagnosis}</b></h3>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
