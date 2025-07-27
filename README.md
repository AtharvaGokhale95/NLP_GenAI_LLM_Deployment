# NLP_GenAI_LLM_Deployment
This repository includes all the basics to advanced level of details related to NLP models, Deep Learning models, Fine Tuning of LLMs, GenAI apps in AWS and Deploying end-to-end projects

 ⁠Ways to command virtual Environment:
    a. Using python command: python -m venv myenv - Activate the venv: source myenv/bin/activate - Deactivate: deactivate
    b. Installing virtualenv: virtualenv -p python3.12 myenv - Activate: source myenv/bin/activate - Deactivate: deactivate
    c. Using Anaconda: 
        1. Created in that specific folder: conda create -p myenv python==3.12 -y - Activate: conda activate ./myenv
        2. Created in Conda's default envs folder: conda create -n myenv python==3.12 -y - Activate: conda activate myenv (No path req)


Machine Learning - NLP:
1. Basic category of ML:
    a. Supervised: 
        1. Categories:
            a. Classification
            b. Regression
        2. Features:
            a. We have Independent features that are I/P features and then we have Dependent features that are O/P features
            b. Types of features/ Variables:
                1. Binary 
                2. Categorical
                3. Continuous
        3. NLP (When the features are in Languages): Independent Features are converted into vectors which provides meaningful information to the model
        4. Flow for NLP Models: 
            1. Text Pre-Processing: Step 1 - Data Cleaning
                a. Tokenization
                b. Lemmatization
                c. Stemming
                d. Stop Words
            2. Text Pre-Processing: Step 2 - Convert I/P text to vectors
                a. Bag of Words
                b. TF-IDF
                c. Unigrams
                d. Bi-Grams
            3. Text Pre-Processing : Step 3 - Convert I/P text to vectors
                a. Word2Vec
                b. Average Word2Vec
            4. Text Pre-Processing : Step 4 - Create word embedding by converting I/P text to vectors - Extension of Step 3
                1. Static Word Embeddings
                    a. FastText
                    b. GloVe
                2. Contextual Word Embeddings
                    a. ELMo (Deep Bi-Directional LSTM)
                    b. BERT
                    c. GPT
                    d. RoBERTa
                3. Sentence and Document Embedding:
                    a. USE (Universal Sentence Encoder)
                    b. InferSent
                    c. SBERT (Sentence BERT)
            5. Deep Learning Models:
                a. RNN
                b. LSTM
                c. GRU RNN
            6. Transformer: BERT
        5. Libraries Used:
            1. ML Libraries: NLTK, Spacy
            2. Deep Learning: TensorFlow, PyTorch


    b. Unsupervised:
        1. 