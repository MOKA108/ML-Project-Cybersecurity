import streamlit as st
import pandas as pd


# the title of the app
# Streamlit UI
st.title("Cybersecurity Attack Type Prediction ML Model")
st.header("CSV File Importer")
st.write("Upload a CSV file to predict the attack type.")


# Set the title of the app
st.write("CSV File Importer")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.write("DataFrame:")
    st.dataframe(df)



