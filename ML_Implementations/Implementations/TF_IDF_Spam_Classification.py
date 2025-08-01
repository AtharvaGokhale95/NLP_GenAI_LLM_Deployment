import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK stopwords are available
nltk.download('stopwords')

# 1. Pre-process single sentence: text will be a string containing a sentence
def preprocess_text(text):
    text = text.lower()                 # Convert to lower case
    text = text.translate(str.maketrans('', '', string.punctuation))    # Remove all the punctuations
    text = re.sub(r'\d+', '', text)     # Replace entire sequence of digits with nothing (remove them)
    text = re.sub('[^a-zA-Z]', " ", text)   # Replace everything apart from a - z and A - Z with a space character
    words = text.split()     # Split the sentence into individual words separated by space character
    stop_words = set(stopwords.words('english'))     # Get a unique set of english stopwords
    words = [word for word in words if word.isalpha() and word not in stop_words]   # Inline function to iterate on every word to remove stopwords
    lemmatizer = WordNetLemmatizer()        # Created an instance of WordNetLemmatizer function
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatized all the words
    return " ".join(words)      # Joined all the words with a space character in between to from a sentence

# 2. Load and preprocess the dataset
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, encoding= 'ISO8859-1')[['v1', 'v2']] # Read the first columns from the dataframe
    df.columns = ['label', 'message']   # Rename the columns
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})    # Map the label values to 0 and 1
    df['clean_message'] = df['message'].apply(preprocess_text)
    # Creates a new column 'clean_message' to store the processed text
    # Apply method let's you apply a function to each element of the df
    return df
    
# 3. Vectorize messages using TF-IDF (Feature Extraction)
def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))  # Created an instance of TfidfVectorizer function
    X_train = vectorizer.fit_transform(train_texts)
    # We fit vectorizer on train_texts dataset to generate X_train which are numerical vectors - Learn the structure of the text
    X_test = vectorizer.transform(test_texts)
    # Convert the test_texts to vectors using the vocabulary learnt during fit()
    return X_train, X_test, vectorizer

# 4. Train a Naive Bayes Classifier
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)  # The Naive Bayes Classifier model is trained on X_train df
    return model        # return the trained model

# 5. Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Accuracy: \n", accuracy_score(y_test, y_pred))
    # All the values are printed so no return statement is required
  
# 6. Main runner function  
def run_spam_classifier(csv_path):
    df = load_and_preprocess(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size = 0.2, random_state = 42)
    X_train, X_test, vectorizer = vectorize_text(X_train, X_test)
    # We need pass the df columns for Features and labels
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model, vectorizer
     
# 7. Run the pipeline
if __name__ == '__main__':          # This line controls what code runs when a Python file runs directly as a script - Python code starts running by executing this conditional check
    model, vectorizer = run_spam_classifier('/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/dataset/spam.csv')
# As the if condition gets satisfied, the control goes inside the loop and run the run_spam_classifier and then so on....