import streamlit as st
import pandas as pd
import pickle
import hdbscan
import hashlib
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import matplotlib.pyplot as plt


# Create a directory to store precomputed cluster results
PRECOMPUTED_DIR = "WebApp/precomputed_clusters"
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

# Function to generate a unique hash for a dataset
def get_data_hash(df):
    """Generate a hash for the dataset to check if it has been processed before."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# Function to load precomputed clusters
def load_precomputed_clusters(data_hash):
    """Check if precomputed clusters exist for the dataset."""
    file_path = os.path.join(PRECOMPUTED_DIR, f"{data_hash}.pkl")
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    return None

# Function to save computed clusters
def save_precomputed_clusters(df, data_hash):
    """Save cluster assignments to avoid recomputation."""
    file_path = os.path.join(PRECOMPUTED_DIR, f"{data_hash}.pkl")
    df.to_pickle(file_path)



# PRESENTATION PART
st.title("Cybersecurity Attack Type Prediction ML Model")
st.header("Upload your CSV file to know the best model for predicting the attack type")

# File uploader widget for the user to drop the cvs file
uploaded_file = st.file_uploader(
    "Please make sure to drop your dataset (.csv) already processed with Feature Engineering, encoded and normalized:",
    type="csv"
)


# Checking if a file has been uploaded
if uploaded_file is not None:

    # Reading the CSV file into a DataFrame
    df_cyber_processed = pd.read_csv(uploaded_file)
    st.write("Processed Dataset preview:")
    st.dataframe(df_cyber_processed.head(25))


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
    tab1, tab2, tab4, tab5 = st.tabs([
        "Logistic Regression",
        "K-Nearest Neighbors",
        "HDBSCAN",
        "HDBSCAN & XgBoost"
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


    with tab4:
        st.title("HDBSCAN")

        # Define target and features
        target = 'Attack Type'
        X = df_cyber_processed.drop(columns=[target])  # Features
        y = df_cyber_processed[target]  # Target

        # Generate dataset hash
        data_hash = get_data_hash(X)

        # Check if clusters were already computed for this dataset
        precomputed_data = load_precomputed_clusters(data_hash)

        if precomputed_data is not None:
            df_cyber_processed = precomputed_data
            is_precomputed = True
        else:
            is_precomputed = False

            # Loading trained HDBSCAN model
            with open("WebApp/trained_models/hdbscan_model.pkl", "rb") as model_file:
                hdbscan_model = pickle.load(model_file)

            # Identify categorical and numerical columns
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

            # Encoding categorical features
            label_encoders = {col: LabelEncoder() for col in categorical_cols}
            for col in categorical_cols:
                X[col] = label_encoders[col].fit_transform(X[col])

            # Normalize numerical features
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

            # Convert to numpy array for HDBSCAN
            X_array = X.to_numpy()

            # Predict clusters using the trained model
            clusters = hdbscan_model.fit_predict(X_array)
            df_cyber_processed['Cluster'] = clusters

            # Map clusters to the most common Attack_Type
            cluster_mapping = df_cyber_processed.groupby('Cluster')[target].agg(lambda x: x.mode()[0]).reset_index()
            cluster_mapping.columns = ['Cluster', 'Predicted_Attack_Type']

            # Apply mapping
            cluster_to_attack = dict(zip(cluster_mapping['Cluster'], cluster_mapping['Predicted_Attack_Type']))
            df_cyber_processed['Predicted_Attack_Type'] = df_cyber_processed['Cluster'].map(cluster_to_attack)

            # Save precomputed clusters for future use
            save_precomputed_clusters(df_cyber_processed, data_hash)

        # Model performance evaluation
        st.subheader("Model Performance Metrics")
        accuracy = accuracy_score(df_cyber_processed[target], df_cyber_processed['Predicted_Attack_Type'])
        st.metric("Accuracy", f"{accuracy:.2%}")

        report_dict = classification_report(df_cyber_processed[target], df_cyber_processed['Predicted_Attack_Type'],
                                            output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        with st.expander("Classification Report"):
            st.dataframe(df_report.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))

        # Alert to say if we used precomputed clusters or if we have to recompute clusters
        message_placeholder = st.empty()

        if is_precomputed:
            message_placeholder.success("Using precomputed cluster results!")
        else:
            message_placeholder.warning("Computing clusters... This might take a few moments.")

        st.write("---")


    with tab5:
        st.title("HDBSCAN & XGBoost")
        #
        # # Assume X_train and y_train are already defined
        # # Define selected features for XGBoost
        # selected_features = [f for f in ['Anomaly Category', 'Time of Day', 'City', 'State'] if
        #                      f in X_train.columns]
        #
        # cluster_models = {}
        # cluster_predictions = []
        #
        # # Add the target column ('Attack Type') to the training data with clusters
        # X_train['Attack Type'] = y_train
        #
        # # Iterate over unique clusters in the training dataset
        # for cluster_id in X_train['Cluster'].unique():
        #     if cluster_id == -1:  # Skip noise cluster
        #         continue
        #
        #     # Subset training data for the cluster
        #     cluster_train_data = X_train[X_train['Cluster'] == cluster_id]
        #
        #     # Ensure selected features are available in the data
        #     cluster_features = [feature for feature in selected_features if feature in cluster_train_data.columns]
        #
        #     # Split features and target for training
        #     X_cluster_train = cluster_train_data[cluster_features]
        #     y_cluster_train = cluster_train_data['Attack Type']  # Target column
        #
        #     # Check if there is enough data in the cluster
        #     if len(X_cluster_train) < 8:
        #         st.warning(f"Skipping cluster {cluster_id} due to insufficient data.")
        #         continue
        #
        #     # Encode target labels
        #     label_encoder = LabelEncoder()
        #     y_cluster_train_encoded = label_encoder.fit_transform(y_cluster_train)
        #
        #     # Train XGBoost model for this cluster
        #     model = XGBClassifier(eval_metric='logloss', random_state=42)
        #     model.fit(X_cluster_train, y_cluster_train_encoded)
        #
        #     # Store the trained model and label encoder
        #     cluster_models[cluster_id] = {
        #         'model': model,
        #         'label_encoder': label_encoder
        #     }
        #
        #     # Make predictions on the training data for this cluster
        #     cluster_preds = model.predict(X_cluster_train)
        #
        #     # Decode predictions back to original labels
        #     cluster_preds_decoded = label_encoder.inverse_transform(cluster_preds)
        #     cluster_predictions.extend(zip(cluster_train_data.index, cluster_preds_decoded))
        #
        # # Store predictions as a DataFrame
        # cluster_predictions_df = pd.DataFrame(cluster_predictions, columns=['Index', 'Predicted Attack Type'])
        # cluster_predictions_df.set_index('Index', inplace=True)
        #
        # # Merge predictions back with the original dataset (if needed)
        # X_train_with_predictions = X_train.copy()
        # X_train_with_predictions['Predicted Attack Type'] = cluster_predictions_df['Predicted Attack Type']
        #
        # # Streamlit display of final predictions
        # st.write("Final Predictions:")
        # st.dataframe(X_train_with_predictions[['Cluster', 'Attack Type', 'Predicted Attack Type']].head())
        #
        # # Add predictions back to the DataFrame
        # predictions_dict = dict(cluster_predictions)
        # data['Cluster_Predicted_Attack_Type'] = data.index.map(predictions_dict)
        #
        # # Handle any NaN values in predictions
        # data['Cluster_Predicted_Attack_Type'].fillna(data['Attack Type'].mode()[0], inplace=True)
        #
        # # Evaluate overall accuracy and classification report
        # accuracy = accuracy_score(data['Attack Type'], data['Cluster_Predicted_Attack_Type'])
        # st.write(f"Cluster-Specific Features Overall Accuracy: {accuracy:.5f}")
        #
        # # Display classification report
        # classification_report_output = classification_report(data['Attack Type'], data['Cluster_Predicted_Attack_Type'])
        # st.text("Classification Report:\n" + classification_report_output)
