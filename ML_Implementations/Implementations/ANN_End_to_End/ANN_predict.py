import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

def load_trained_model(filepath):
    model = load_model(filepath)
    return model

def load_pickle_file(filepath_geography, filepath_gender, filepath_scaler):
    with open(filepath_geography, 'rb') as file:
        onehotencoder_geography = pickle.load(file)
    with open(filepath_gender, 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open(filepath_scaler, 'rb') as file:
        scaler = pickle.load(file)
    return onehotencoder_geography, label_encoder_gender, scaler

def preProcess_prediction_data(input_data, onehotencoder_geography,label_encoder_gender, scaler):
    df = pd.DataFrame([input_data])
    df['Gender'] = label_encoder_gender.transform(df['Gender'])
    geography_encoder = onehotencoder_geography.transform([[input_data['Geography']]]).toarray()
    geography_encoded_df = pd.DataFrame(geography_encoder, columns = onehotencoder_geography.get_feature_names_out(['Geography']))
    df = pd.concat([df.drop('Geography', axis= 1), geography_encoded_df], axis= 1)
    df = scaler.transform(df)
    return df

def predict_model(df, model):
    prediction = model.predict(df)
    prediction_probability = prediction[0][0]
    if prediction_probability > 0.5:
        print("Customer will be churned")
    else:
        print("Customer will not be churned")
    return prediction_probability

def run_ann(filepath, input_data):
    model = load_trained_model(filepath)
    onehotencoder_geography, label_encoder_gender, scaler = load_pickle_file(
        '/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/Implementations/ANN_End_to_End/onehotencoder_geography.pkl',
        '/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/Implementations/ANN_End_to_End/label_encoder_gender.pkl',
        '/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/Implementations/ANN_End_to_End/scaler.pkl'
    )
    df = preProcess_prediction_data(input_data, onehotencoder_geography, label_encoder_gender, scaler)
    prediction_probability = predict_model(df, model)
    return prediction_probability

     
if __name__ == '__main__':
    prediction_probability = run_ann(
        '/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/Implementations/ANN_End_to_End/model.h5',
        input_data = {
            'CreditScore': 600,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 40,
            'Tenure': 3,
            'Balance': 60000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 50000
            }
        )
    
    
    
    
    