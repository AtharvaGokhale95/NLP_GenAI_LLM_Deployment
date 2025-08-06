import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import gensim.downloader as api
import pandas as pd
import numpy as np
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK stopwords are available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# 1. Pre-process single sentence: text will be a string containing a sentence
def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub('[^a-zA-Z]', " ", text)  # Replace non-alphabetic characters with space
    words = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Set of stopwords
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alpha
    return words  # Return list of tokens

# 2. Load and preprocess the dataset
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, encoding='ISO8859-1')[['v1', 'v2']]  # Read dataset and select columns
    df.columns = ['label', 'message']  # Rename columns
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1
    df['tokens'] = df['message'].apply(preprocess_text)  # Apply preprocessing to messages
    return df

# 3. Train Word2Vec model on tokenized text
def train_word2vec_model(token_lists, vector_size=100, window=5):
    model = Word2Vec(sentences=token_lists, vector_size=vector_size, window=window, min_count=1, workers=4)
    return model

# 4. Convert each message into a fixed-size vector using Word2Vec embeddings
def vectorize_text_with_word2vec(token_lists, w2v_model, vector_size=100):
    def vectorize(tokens):
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)
    return np.vstack([vectorize(tokens) for tokens in token_lists])

# 5. Train a classifier (Random Forest)
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 6. Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))

# 7. Main runner function  
def run_spam_classifier(csv_path):
    df = load_and_preprocess(csv_path)  # Step 1: Load and preprocess
    w2v_model = train_word2vec_model(df['tokens'])  # Step 2: Train Word2Vec model
    features = vectorize_text_with_word2vec(df['tokens'], w2v_model)  # Step 3: Convert text to vectors
    X_train, X_test, y_train, y_test = train_test_split(features, df['label'], test_size=0.2, random_state=42)  # Step 4: Split data
    model = train_model(X_train, y_train)  # Step 5: Train model
    evaluate_model(model, X_test, y_test)  # Step 6: Evaluate
    return model, w2v_model

# 8. Run the pipeline
if __name__ == '__main__':  # Ensures code runs only when the script is executed directly
    model, word2vec_model = run_spam_classifier('/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/dataset/spam.csv')