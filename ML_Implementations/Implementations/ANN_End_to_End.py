import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load and preprocess the dataset
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, encoding= 'ISO8859-1')
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)      # Drop the irrelevant columns from the dataframe
    label_encoder_gender = LabelEncoder()                               # Instance of the class LabelEncoder - Used for binary categories
    df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])     # This converts Male and Female to 1 and 0
    onehotencoder_geography = OneHotEncoder()                           # Instance of class OneHotEncoder - Convert categorical to numerical values
    geography_encoder = onehotencoder_geography.fit_transform(df[['Geography']])        # 2-D array is expected 
    df_geography = pd.DataFrame(geography_encoder.toarray(), columns=onehotencoder_geography.get_feature_names_out(['Geography']))
    df = pd.concat([df.drop('Geography', axis=1), df_geography], axis = 1)
    return df


    
def divide_split_scale_data(df):
    # Split data into dependent (y) and independent features (X)
    X = df.drop('Exited', axis = 1)
    y = df['Excited']
    # Split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale the features: It transforms your data so that each feature has a mean of 0 and standard deviation of 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
def create_pickle_file(label_encoder_gender, onehotencoder_geography, scaler):
    with open('label_encoder_gender.pkl', 'wb') as file:                # Open in write-byte mode
        pickle.dump(label_encoder_gender, file)
    with open('onehotencoder_geography.pkl', 'wb') as file:
        pickle.dump(onehotencoder_geography, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

dataframe = load_and_preprocess('/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/dataset/Churn_Modelling.csv')
print(dataframe)