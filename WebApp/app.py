import streamlit as st
import pandas as pd
import pickle
import hdbscan

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import matplotlib.pyplot as plt


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
    st.write("Dataset preview:")
    st.dataframe(df_cyber_processed)


    # TRAINING PART

    # we drop 'User Information' column from the dataset
    df_cyber_processed = df_cyber_processed.drop(columns=['User Information'], errors='ignore')

    # Separate the features and the target variable.
    # 'Attack Type' is our target, so we drop it from features.
    X = df_cyber_processed.drop('Attack Type', axis=1)  # Features: all columns except 'Attack Type'
    y = df_cyber_processed['Attack Type']  # Target: the 'Attack Type' column

    # For the 3 classic models
    # we split the data into training and testing sets (test_size=0.2 means 20% of the data is used for testing)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42)

    # For the 2 cluster models
    # we split the data into train, test, and validation sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                                test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.25, random_state=42,stratify=y_train_val)



    # MODELING PART

    # creation of tabs for each model
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "HDBSCAN",
        "model 5"
    ])

    with tab1:
        st.title("Logistic Regression")

        # Load the trained model from a pickle file
        model_filename = 'WebApp/trained_models/logistic_regression_model.pkl'  # Update this path to your .pkl file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict on the test data
        y_pred = model.predict(X_test_1)

        # Calculate accuracy
        acc = accuracy_score(y_test_1, y_pred)

        st.subheader("Model Performance Metrics")

        # Display model performance metrics
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test_1, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report:"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        st.write("---")

    with tab2:
        st.title("K-Nearest Neighbors")

        # Load the trained model from a pickle file
        model_filename = 'WebApp/trained_models/k-nearest_neighbors_model.pkl'  # Update this path to your .pkl file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict on the test data
        y_pred = model.predict(X_test_1)

        # Calculate accuracy
        acc = accuracy_score(y_test_1, y_pred)

        st.subheader("Model Performance Metrics")

        # Display model performance metrics
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test_1, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report:"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        st.write("---")


    with tab3:
        st.title("Support Vector Machine")

        # Load the trained model from a pickle file
        model_filename = 'WebApp/trained_models/support_vector_machine_model.pkl'  # Update this path to your .pkl file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict on the test data
        y_pred = model.predict(X_test_1)

        # Calculate accuracy
        acc = accuracy_score(y_test_1, y_pred)

        st.subheader("Model Performance Metrics")

        # Display model performance metrics
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test_1, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report:"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        st.write("---")



    with tab4:
        st.title("HDBSCAN")

        # Load trained HDBSCAN model
        with open("WebApp/trained_models/hdbscan_model.pkl", "rb") as model_file:
            hdbscan_model = pickle.load(model_file)

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

        # Predict clusters using the trained model
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

        st.write("---")


    with tab5:
        st.text("BEST CLUSTER MODEL : HDBSCAN+XBOOST")
