def setup_nlp_environment():
    import pandas as pd
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Initialize stemmer
    PS = PorterStemmer()
    
    # Initialize CountVectorizer
    cv = CountVectorizer(max_features=100, binary=True)
    
    return {
        'pd': pd,
        're':re,
        'nltk': nltk,
        'stopwords': stopwords,
        'PS': PS,
        'cv': cv
    }

def read_csv_file(path_to_data):
    messages = env['pd'].read_csv(path_to_data, sep = ',', encoding='latin1', names=['Label', 'Message', 'c1', 'c2', 'c3'], header = 0)
    # names attribute renames the columns and header tells panda to skip the first row as column names and use the new column names
    messages = messages.drop(messages.columns[-3 : ], axis = 1)
    return messages

def data_preprocessing(messages):
    corpus = []                                                                 # Create a empty list in which we can store the processed text     
    for idx in range(0, len(messages)):
        review = env['re'].sub('[^a-zA-Z]', ' ', messages['Message'][idx])      # Anything apart from a - z and A - Z, replace with " ". Only keep the characters
        # review is a string which contains a sentence which will be appended as a element in the list corpus
        review = review.split()                                                 # Split a sentence into words
        review = [env['PS'].stem(word) for word in review if word.lower() not in env['stopwords'].words('english')]
        # 1. Convert all the words in lower case
        # 2. Remove the stopwords
        # 3. Apply Porter Stemmer
        review = ' '.join(review)                                               # Joins the words separated by " " to again form a sentence
        corpus.append(review)
    return corpus

def BOW_text_to_vector(corpus):
    X = env['cv'].fit('corpus').toArray()
    

env = setup_nlp_environment()
messages = read_csv_file('/Users/atharva/Documents/GitHub/NLP_GenAI_LLM_Deployment/ML_Implementations/dataset/spam.csv')
corpus = data_preprocessing(messages)
print(corpus[0:5])