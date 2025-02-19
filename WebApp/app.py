import streamlit as st
import pandas as pd
import pickle
import hdbscan

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    st.write("Dataset preview:")
    st.dataframe(df_cyber_processed)


    # TRAINING PART

    # Drop 'User Information' column from the dataset
    df_cyber_processed = df_cyber_processed.drop(columns=['User Information'], errors='ignore')

    # Separate the features and the target variable.
    # 'Attack Type' is our target, so we drop it from features.
    X = df_cyber_processed.drop('Attack Type', axis=1)  # Features: all columns except 'Attack Type'
    y = df_cyber_processed['Attack Type']  # Target: the 'Attack Type' column

    # Split the data into training and testing sets.
    # test_size=0.3 means 30% of the data is used for testing.
    # random_state ensures the split is reproducible.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # # Print the shapes of the resulting datasets to confirm the split.
    # st.write("\nData Splitting Results:")
    # print("Training set shape (features):", X_train.shape)
    # print("Training set shape (target):", y_train.shape)
    # print("Testing set shape (features):", X_test.shape)
    # print("Testing set shape (target):", y_test.shape)


    # MODELING PART

    # creation of tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Support Vector Machine",
        "model 4",
        "model 5"
    ])

    with tab1:
        st.title("Logistic Regression")

        # Load the trained model from a pickle file
        model_filename = 'WebApp/trained_models/logistic_regression_model.pkl'  # Update this path to your .pkl file
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Model Performance Metrics")

        # Display model performance metrics
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test, y_pred, output_dict=True)
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
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Model Performance Metrics")

        # Display model performance metrics
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test, y_pred, output_dict=True)
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
        y_pred = model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Model Performance Metrics")

        # Display model performance metrics
        st.metric(label="Accuracy", value=f"{acc:.2%}")

        # Show classification report as a DataFrame
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report:"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        st.write("---")



    with tab4:
        st.text("model 4 for CLUSTER")


    with tab5:
        st.text("BEST CLUSTER MODEL : HDBSCAN")
