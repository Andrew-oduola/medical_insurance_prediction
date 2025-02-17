import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('insurance_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict car price
def insurance_prediction(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Medical Insurance Prediction", page_icon="💸", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("💸 Medical Insurance Prediction")
    st.markdown("This app predicts the cost of medical insurance based on the information provided.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a machine learning model to predict the cost of medical insurance based on the information provided. The model was trained on the [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, value=63, help="Enter the age")
        sex = st.selectbox("Gender", ["Male", "Female"], help="Select the sex")
        bmi = st.number_input("BMI", min_value=0, value=62, help="Enter the BMI")
        
    with col2:
        children = st.number_input("Children", min_value=0, value=0, help="Enter the number of children")
        smoker = st.selectbox("Smoker", ["Yes", "No"], help="Select if the person is a smoker")
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"], help="Select the region")
       
    # Prepare input data for the model
    sex = 0 if sex == "Male" else 1
    smoker = 0 if smoker == "Yes" else 1
    region = 0 if region == "Southeast" else 1 if region == "Southwest" else 2 if region == "Northeast" else 3


    # Prepare input data for the model
    input_data = [age, sex, bmi, children, smoker, region]
    # input_data = [float(x) for x in input_data]  # Ensure all data is float

    # Prediction button
    if st.button("Predict"):
        try:
            prediction = insurance_prediction(input_data)
            
            if prediction[0] > 0:
                prediction_text = f"The predicted cost of medical insurance is {prediction[0]:.2f}"
                result_placeholder.success(prediction_text)
                st.success(prediction_text)
            else:
                prediction_text = f"An error occurred: change some information"
                result_placeholder.error(prediction_text)
                st.error(prediction_text)
            
            
            # st.markdown("**Note:** This is a simplified model and may not be accurate for all cases.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()