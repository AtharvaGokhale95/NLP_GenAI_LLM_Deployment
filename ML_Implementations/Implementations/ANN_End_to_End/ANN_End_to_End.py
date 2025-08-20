import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

# 1. Load and preprocess the dataset
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, encoding= 'ISO8859-1')
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)                      # Drop the irrelevant columns from the dataframe
    label_encoder_gender = LabelEncoder()                                               # Instance of the class LabelEncoder - Used for binary categories
    df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])                     # This converts Male and Female to 1 and 0
    onehotencoder_geography = OneHotEncoder()                                           # Instance of class OneHotEncoder - Convert categorical to numerical values
    geography_encoder = onehotencoder_geography.fit_transform(df[['Geography']])        # 2-D array is expected 
    # Create a df with column names returned by the get_feature_names_out method and the values from geography_encoder but converted to array 
    df_geography = pd.DataFrame(geography_encoder.toarray(), columns=onehotencoder_geography.get_feature_names_out(['Geography']))
    df = pd.concat([df.drop('Geography', axis=1), df_geography], axis = 1)              # Concatenate the original df and df_geography
    return df, label_encoder_gender, onehotencoder_geography

# 2. Convert the df into dependent and independent features - Split the data into training and test data - Scale the Independent Features
def divide_split_scale_data(df):
    # Split data into dependent (y) and independent features (X)
    X = df.drop('Exited', axis = 1)
    y = df['Exited']
    # Split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale the features: It transforms your data so that each feature has a mean of 0 and standard deviation of 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler         # Returning the trainer scaler object containing the mean/std values for the training data            

# 3. Saving the trained label encoder, one hot encoder and the scaler objects
def create_pickle_file(label_encoder_gender, onehotencoder_geography, scaler):
    with open('label_encoder_gender.pkl', 'wb') as file:                # Open in write-byte mode
        pickle.dump(label_encoder_gender, file)
    with open('onehotencoder_geography.pkl', 'wb') as file:
        pickle.dump(onehotencoder_geography, file)
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

# 4. Definition of the Sequential ANN model
def ann_model_definition(X_train):
    model=Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Details for the first hidden layer 
        # 64 neurons, activation function for each neuron = relu, as this is the first hidden layer, we mention the input shape = X_train.shape
        # X_train.shape: Returns (rows [0], columns [1]), we choose X_train.shape[1] as we only need the no of independent/ input features - Columns
        Dense(32, activation='relu'),                                   # Details for second hidden layer
        Dense(1, activation='sigmoid')                                  # Details for output layer
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)            # Define the optimizer
    loss = tf.keras.losses.BinaryCrossentropy()                         # Define the loss function
    model.compile(optimizer=opt, loss = loss, metrics = ['accuracy']) # Compile the model with the defined optimizer, loss function and performance metric
    return model

# 5. Setup the Tensorboard and define the early stopping conditions
def setup_tensorboard_earlystopping():
    #Set up Tensorboard:
    log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorflow_callback = TensorBoard(log_dir, histogram_freq=1)
    # Setup Early Stopping:
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    return tensorflow_callback, early_stopping_callback

# 6. Train the model on the dataset for 100 epochs
def train_model(model, X_train, y_train, X_test, y_test, tensorflow_callback, early_stopping_callback):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs = 100,
        callbacks = [tensorflow_callback, early_stopping_callback]
    )
    return model

# 7. 
def save_model(model):
    model.save('model.h5')

        
def run_ann(filepath):
    df, label_encoder_gender, onehotencoder_geography = load_and_preprocess(filepath)
    X_train, X_test, y_train, y_test, scaler = divide_split_scale_data(df)
    create_pickle_file(label_encoder_gender, onehotencoder_geography, scaler)
    model = ann_model_definition(X_train)
    tensorflow_callback, early_stopping_callback = setup_tensorboard_earlystopping()
    model = train_model(model, X_train, y_train, X_test, y_test, tensorflow_callback, early_stopping_callback)
    save_model(model)
    return model
    
     
if __name__ == '__main__':
    model = run_ann('/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/dataset/Churn_Modelling.csv')