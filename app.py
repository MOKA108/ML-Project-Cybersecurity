import streamlit as st
import pandas as pd
import pickle
# import hdbscan
# from sklearn.metrics import classification_report
#
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import classification_report, accuracy_score


# creating the title of the app (Streamlit UI)
st.title("Cybersecurity Attack Type Prediction ML Model")
st.header("CSV File Importer")

# Creating a file uploader widget
uploaded_file = st.file_uploader("Upload a CSV file to predict the attack type.", type="csv")

# Loading the trained model
model = pickle.load(open('trained_hdbscan_model.pkl', 'rb'))

# Checking if a file has been uploaded
if uploaded_file is not None:
    # Reading the CSV file into a DataFrame
    data_encoded = pd.read_csv(uploaded_file)
    
    # Displaying the DataFrame
    st.write("DataFrame:")
    st.dataframe(data_encoded)

    # Making predictions
    predictions = model.fit_predict(data_encoded)

    # Show cluster assignments
    st.subheader("Cluster Assignments")
    st.write(pd.DataFrame(predictions, columns=["Cluster"]))
