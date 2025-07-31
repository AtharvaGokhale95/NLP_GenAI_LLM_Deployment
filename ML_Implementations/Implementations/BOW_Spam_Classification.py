import pandas as pd
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Ensure NLTK stopwords are available
nltk.download('stopwords')

# 1. Preprocess a single message
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub('[^a-zA-Z]', " ", text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# 2. Load and preprocess the dataset
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean_message'] = df['message'].apply(preprocess_text)
    return df

# 3. Vectorize messages using Bag of Words
def vectorize_text(train_texts, test_texts):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# 4. Train a Naive Bayes classifier
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# 5. Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Main runner function
def run_spam_classifier(csv_path):
    df = load_and_preprocess(csv_path)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.2, random_state=42
    )
    X_train, X_test, vectorizer = vectorize_text(X_train_text, X_test_text)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model, vectorizer

# 7. Run the pipeline
if __name__ == '__main__':
    model, vectorizer = run_spam_classifier('/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/dataset/spam.csv')















