import streamlit as st
import pandas as pd
# import pickle
import hdbscan

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# import numpy as np
# import matplotlib.pyplot as plt


# PRESENTATION PART

# creating the title of the app (Streamlit UI)
st.title("Cybersecurity Attack Type Prediction ML Model")
st.header("CSV File Importer")

# Creating a file uploader widget
uploaded_file = st.file_uploader("Upload a CSV file to predict the attack type.", type="csv")

# Loading the trained model
# model = pickle.load(open('trained_hdbscan_model.pkl', 'rb'))

# Checking if a file has been uploaded
if uploaded_file is not None:
    # Reading the CSV file into a DataFrame
    df_cyber_processed = pd.read_csv(uploaded_file)

    # Displaying the DataFrame
    st.write("DataFrame:")
    st.dataframe(df_cyber_processed)

    # Making predictions
    # predictions = model.fit_predict(df_cyber_processed)

    # Show cluster assignments
    # st.subheader("Cluster Assignments")
    # st.write(pd.DataFrame(predictions, columns=["Cluster"]))



    # MODELING PART

    # creation of tabs
    tab1, tab2, tab3 = st.tabs(["HDBSCAN", "model 2", "model 3"])

    with tab1:

        # Define the target variable and features
        target = 'Attack Type'
        features = df_cyber_processed.drop(columns=[target])

        # Identify categorical and numerical columns
        categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = features.select_dtypes(exclude=['object']).columns.tolist()

        # Encoding categorical features using Label Encoding
        label_encoders = {col: LabelEncoder() for col in categorical_cols}
        for col in categorical_cols:
            features[col] = label_encoders[col].fit_transform(features[col])

        # Normalizing numerical features using StandardScaler
        scaler = StandardScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

        # Convert features to numpy array for HDBSCAN
        X = features.to_numpy()

        # Create an instance of HDBSCAN
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)

        # Fit the model
        clusters = hdbscan_model.fit_predict(X)

        # Add the cluster labels to the original DataFrame
        df_cyber_processed['Cluster'] = clusters

        # Map clusters to the most common Attack_Type
        cluster_mapping = df_cyber_processed.groupby('Cluster')[target].agg(
            lambda x: x.value_counts().index[0]).reset_index()
        cluster_mapping.columns = ['Cluster', 'Predicted_Attack_Type']

        # Create a mapping dictionary
        cluster_to_attack = dict(zip(cluster_mapping['Cluster'], cluster_mapping['Predicted_Attack_Type']))

        # Assign predicted Attack_Type based on clusters
        df_cyber_processed['Predicted_Attack_Type'] = df_cyber_processed['Cluster'].map(cluster_to_attack)

        # RESULT PART

        # Evaluating the model
        st.header("Model Performance Metrics")
        accuracy = accuracy_score(df_cyber_processed[target], df_cyber_processed['Predicted_Attack_Type'])
        st.metric("Accuracy", f"{accuracy:.2%}")

        # Parsing classification report into a DataFrame
        report_dict = classification_report(
            df_cyber_processed[target],
            df_cyber_processed['Predicted_Attack_Type'],
            output_dict=True
        )

        # Converting dictionary to DataFrame
        df_report = pd.DataFrame(report_dict).transpose()

        # Displaying key metrics
        st.subheader("HDBSCAN Classification Report")

        # Displaying report as a table
        with st.expander("View Full Report"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))


    with tab2:
        st.text("model 2")

    with tab3:
        st.text("model 3")
