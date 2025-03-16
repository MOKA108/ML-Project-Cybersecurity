# DSTI Python Machine Learning Project

## Overview
This project is done in the scope of the DSTI Python ML course.     
The goal is to returns the prediction for the type of cyber-attack.     
You can find the main project in the folder "MLproject".      
You can find the app files in the folder "WebApp".    
    
    
Here is the Web Application: https://cybersecurity-attack-type-prediction.streamlit.app/  

## How to use the app:
1) go inside the folder "Data"
2) download the file "Encoded_Cybersecurity_Data.csv"
3) open the app
4) drop the csv file in the dropbox and explore the models


## The analysis includes 4 parts:
- Exploratory Data Analysis
- Feature Engineering
- Encoding and Normalizing
- Modeling


## The project is composed of:
- 3 Jupyter Notebooks to analyse the raw dataset and train the best ML model to predict the Attack Types
  - a notebook for the EDA, FE, encoding and normalizing parts
  - a notebook with the classification models
  - a notebook dedicated to the clusterization models
- a report explaining the whole project and our decisions
- a GitHub page (where you are right now!)
- a Web Application to show result of the model prediction
- a video to show a short presentation of the application


## Here is the team who worked on it:
- 2 Data Scientists: Yasmine El khdar, Gaia Bianciotto
- 1 Data Engineer: Marie-Caroline Bertheau
- 3 Data Analysts: Giti Shekari, Aubain Kokou Viglo, Anthony Baudchon


## To go further on this project:
- we can try to optimize the web application by adding the xgboost trained model as a pickle file    
- we can add the last model which correspond to HDBSCAN with XGBoost and with the hyperparameters found thanks to Optuna    
- we can also add a menu with 2 options: the first page would be the one with all the model prediction and precision (already done)      
- on the other page of the menu, we can make the user select input as "device", "location"...etc (from the top features) and get an output for the predicted type of attack (DDoS, Intrusion or Malware)      
- we can eventually add an input in the second page for choosing a model (or adding only the best model for this page)
- about the encoded dataset to drop in the drop box, we can implement the EDA, FE and Encoding parts in the app.py to be able to drop a raw dataset instead       
    
    
To reproduce the project, you can use the requirements.txt file (located in the folder "WebApp") to install the necessary packages.    
